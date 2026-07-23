"""Multi-agent proof loop: planner -> prover -> two verifiers -> DAG.

Design note on thinking vs. structured output
---------------------------------------------
Ollama's `format` parameter constrains generation with a GBNF grammar. That
grammar zeroes the probability of the <think> token, so passing `format`
silently disables reasoning. Passing `think: false` is worse: on models whose
chat template uses think tokens (qwen3.x, gemma4) it makes `format` be ignored
entirely, and you get plain prose back instead of JSON.

So we never combine the two, and we never send `think: false`:

  * Reasoning stage  - no `format`. The model thinks freely; Ollama returns the
                       chain of thought in `message.thinking`, separate from
                       `message.content`.
  * Extraction stage - `format` set to an explicit JSON schema, applied to the
                       *text the reasoning stage already produced*. No new
                       mathematics happens here, so losing thinking costs
                       nothing.

All four agent prompts (planner.md, prover.md, verifier_a.md, verifier_b.md)
instruct the model to emit strict JSON directly, so each response is first
parsed as-is; the extraction stage only runs as a fallback when the model
fails to comply. The prover never round-trips through a second model call at
all: its JSON is parsed locally and, failing that, the reasoning content is
taken verbatim, so the proof text can't be abridged or paraphrased.

Usage
-----
    python main.py                       # verbose, defaults
    python main.py --no-verbose          # silent
    python main.py --max-iterations 25
    python main.py --model qwen3.6:27b --conjecture other.md
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

# ----------------------------------------------------------------------------
# Configuration (all overridable from the command line; see main())
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
# qwen3.6:35b — MoE, 36B total / ~3B active (a3b), Q4_K_M, ~24GB, 256K ctx,
# arch qwen35moe. Same blob (07d35212591f) as :latest, :35b-a3b and
# :35b-a3b-q4_K_M. NOT the same as :35b-mlx (Apple Silicon) or :35b-a3b-mtp-*.
# Same template family as qwen3.5, so `think: false` would break `format`
# here; see the module docstring.
MODEL_NAME = "qwen3.6:35b"
CONJECTURE_FILE = "conjecture.md"
DAG_FILE = "dag.json"
MAX_ITERATIONS = 10
LLM_MAX_RETRIES = 3
REQUEST_TIMEOUT = 1800       # Thinking models are slow; give them room

# Ollama defaults num_ctx low (2048/4096) regardless of model capability.
# Reasoning tokens are drawn from the same budget as the answer, so a thinking
# prover needs a much larger allowance than a one-shot one.
NUM_CTX = 40960              # model supports 256K; raise if VRAM allows
# Per-role generation budgets. Thinking and answer share this stream, so the
# prover — which must think *and* then write a full proof — needs the most.
# reason() grows these on truncation, clamped to whatever num_ctx allows.
NUM_PREDICT_REASONING = {
    "planner": 8192,
    "prover": 24576,
    "verifier": 12288,
}
NUM_PREDICT_EXTRACT = 1024

# Thinking level: True, or "low"/"medium"/"high"/"max" on models that support
# levels. Set to None to omit the field entirely (the model's default).
# Never set this to False — see the module docstring.
THINK = {
    "planner": True,
    "prover": True,
    "verifier": True,
}

# Sampling. qwen3.6's Modelfile defaults are min_p 0, presence_penalty 1.5,
# repeat_penalty 1, temperature 1. The presence penalty is the problem: it
# pushes the model off tokens it has already used. That suppresses repetition
# loops in agentic coding, but mathematics *requires* hammering the same
# symbols (\epsilon, n, x_i) over and over, and the extraction stage's whole
# job is verbatim copying. Left at 1.5 the extractor paraphrases your proof.
# So we damp it for reasoning and switch it off entirely for extraction.
REASONING_OPTIONS: Dict[str, Any] = {
    "temperature": 0.7,          # Greedy decoding degrades thinking models.
    "presence_penalty": 0.4,     # Down from 1.5: proofs reuse notation.
    "min_p": 0.0,
    "num_ctx": NUM_CTX,
    # num_predict is set per role by reason(); see NUM_PREDICT_REASONING.
}

EXTRACT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,          # Mechanical, and grammar-constrained anyway.
    "presence_penalty": 0.0,     # Must be 0: this stage copies, it doesn't write.
    "min_p": 0.0,
    "num_ctx": NUM_CTX,
    "num_predict": NUM_PREDICT_EXTRACT,
}

# Per-role reasoning temperature overrides.
TEMPERATURES = {
    "planner": 0.7,
    "prover": 0.6,     # Slightly tighter: rigour over exploration.
    "verifier": 0.8,   # Looser, so the two reviews don't collapse into one.
}

# Set at runtime if schema-constrained extraction proves incompatible with the
# model's default thinking (empty content, output stranded in .thinking).
_SCHEMA_MODE_BROKEN = False


def log(message: str, verbose: bool = True) -> None:
    """Single logging chokepoint, so --no-verbose silences everything."""
    if verbose:
        print(message)


# ----------------------------------------------------------------------------
# Response schemas (Ollama builds a grammar from these, so required keys and
# enum values are guaranteed rather than hoped for).
# ----------------------------------------------------------------------------
PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_conjecture_proved": {"type": "boolean"},
        "plan_summary": {"type": "string"},
        "next_lemma": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "statement": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "statement", "dependencies"],
        },
    },
    "required": ["is_conjecture_proved", "plan_summary", "next_lemma"],
}

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "justification": {"type": "string"},
    },
    "required": ["decision", "justification"],
}


# ----------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------
def load_file(filepath: str) -> str:
    """Read prompt or markdown file contents."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_dag() -> Dict[str, Any]:
    """Load current DAG state from file. Absent file => empty DAG."""
    if os.path.exists(DAG_FILE):
        with open(DAG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lemmas": {}}


def save_dag(dag: Dict[str, Any]) -> None:
    """Save updated DAG state to file, creating it if needed."""
    with open(DAG_FILE, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2)


# ----------------------------------------------------------------------------
# JSON repair (fallback path; the schema grammar should make it unnecessary)
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_HEX = set("0123456789abcdefABCDEF")


def _repair_escapes(text: str) -> str:
    """Fix backslash escapes inside JSON strings, without corrupting text
    that is already correctly escaped.

    A regex can't do this: in `\\\\subset` (a valid escaped backslash followed
    by 's'), a lookahead-based pattern matches the *second* backslash of the
    pair and doubles it, turning valid JSON into an invalid `\\s` escape.
    The scanner consumes `\\\\` pairs atomically so that can't happen.

    Heuristics inside strings:
      \\ + \\            -> valid pair, keep
      \\ + " or /        -> valid escape, keep
      \\u + 4 hex digits -> valid unicode escape, keep
      \\ + bfnrtu + [a-z]-> LaTeX command (\\frac, \\neq, \\to), double it
      \\ + bfnrtu        -> genuine control escape, keep
      \\ + anything else -> invalid (\\alpha, \\subset, \\{), double it
    """
    out: List[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        nxt = text[i + 1] if i + 1 < n else ""
        if nxt == "\\":
            out.append("\\\\")          # valid pair — consume both, untouchable
            i += 2
        elif nxt in '"/':
            out.append("\\" + nxt)      # valid escape
            i += 2
        elif nxt == "u" and set(text[i + 2 : i + 6]) <= _HEX and len(text[i + 2 : i + 6]) == 4:
            out.append(text[i : i + 6])  # valid \uXXXX
            i += 6
        elif nxt in "bfnrtu":
            follow = text[i + 2 : i + 3]
            if follow.islower() and follow.isalpha():
                out.append("\\\\" + nxt)  # \frac, \neq, \to — LaTeX, not control
            else:
                out.append("\\" + nxt)    # genuine \n, \t, ...
            i += 2
        elif nxt == "":
            out.append("\\\\")          # trailing backslash at end of text
            i += 1
        else:
            out.append("\\\\" + nxt)    # \alpha, \subset, \{ — invalid, double
            i += 2
    return "".join(out)


def _extract_json_object(text: str) -> str:
    """First balanced {...} span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced => truncated


def _salvage_truncated(text: str) -> str:
    """Close an unterminated string and any open braces, so a cut-off reply is
    still usable instead of lost."""
    in_str, esc, depth = False, False, 0
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    repaired = text.rstrip()
    if in_str:
        repaired = repaired.rstrip("\\") + '"'
    return repaired + "}" * max(depth, 0)


def clean_json_text(raw: str) -> str:
    """Normalise model output into parseable JSON.

    Repairs escalate and each is tried only if the previous stage fails to
    parse — text that is already valid JSON is returned byte-for-byte
    untouched, so the repair heuristics can never corrupt a correct reply.
    """
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text).strip()
    text = _extract_json_object(text)

    candidates = [text]
    repaired = _repair_escapes(text)
    if repaired != text:
        candidates.append(repaired)
    candidates.append(_salvage_truncated(repaired))

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return candidates[-1]  # let the caller surface the real parse error


def parse_json_or_none(text: str) -> Optional[Dict[str, Any]]:
    """Parse an agent reply that already complies with its prompt file.

    planner.md, prover.md, verifier_a.md and verifier_b.md all end with
    "Output strictly valid JSON ... no markdown fences and no extra text", so
    the reasoning stage's content is usually the JSON object itself. When it
    parses, we use it directly and skip the extraction call entirely.
    """
    if not text:
        return None
    try:
        obj = json.loads(clean_json_text(text))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _headroom(payload: Dict[str, Any]) -> int:
    """Tokens available for generation after the prompt, with a safety margin.

    num_ctx caps prompt + generation together, so doubling num_predict past
    this point buys nothing: the model would be cut off by the context window
    instead of by the budget, after a long generation.
    """
    chars = sum(len(m["content"]) for m in payload["messages"])
    return max(payload["options"]["num_ctx"] - chars // 4 - 512, 0)


def reason(
    system_prompt: str,
    user_prompt: str,
    role: str,
    think: Any = True,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, status). status is "" on success, or "ceiling" when the
    role exhausted the context window without finishing — a signal that the
    task is too large, not that the call failed.

    Truncated output is never returned as if it were complete: `done_reason ==
    "length"` means the model was cut off mid-sentence, so we grow the budget
    and retry rather than shipping a stump downstream.
    """
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            **REASONING_OPTIONS,
            "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
            "num_predict": NUM_PREDICT_REASONING.get(role, 12288),
        },
    }
    # Only send `think` when we actually want it on. Never send False.
    if think is not None and think is not False:
        payload["think"] = think

    last_partial = ""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            body = _post(payload)
            msg = body.get("message", {})
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""

            hit_ceiling = body.get("done_reason") == "length"
            if not thinking and "<think>" in content:
                # Model inlined its reasoning instead of separating it.
                m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if m:
                    thinking = m.group(1).strip()
                content = _THINK_RE.sub("", content).strip()

            if content.strip() and not hit_ceiling:
                return content.strip(), ""

            if hit_ceiling:
                budget = payload["options"]["num_predict"]
                log(
                    f"  ⚠️  {role} was cut off at num_predict={budget} "
                    f"(thinking used ~{len(thinking) // 4} tokens).",
                    verbose,
                )
                if content.strip():
                    last_partial = content.strip()
                room = _headroom(payload)
                new = min(budget * 2, room)
                if new > budget and attempt < LLM_MAX_RETRIES:
                    payload["options"]["num_predict"] = new
                    log(f"     retrying with num_predict={new}.", verbose)
                    continue
                # No headroom left: the window itself is the limit.
                log(
                    f"  ⛔ {role} exhausted the context window "
                    f"(num_ctx={payload['options']['num_ctx']}, room={room}).",
                    verbose,
                )
                return "", "ceiling"

            log(f"  ⚠️  {role} returned empty content (attempt {attempt}).", verbose)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} transport error (attempt {attempt}): {e}", verbose)

    # Retries exhausted. A truncated draft beats nothing, but flag it as such.
    return (last_partial, "ceiling") if last_partial else ("", "")


def extract(
    instruction: str,
    source_text: str,
    schema: Dict[str, Any],
    role: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Stage 2: convert stage-1 prose into schema-conformant JSON.

    `think` is deliberately omitted (not set False) so the grammar constraint
    is honoured on models whose templates use think tokens.
    """
    system = (
        "You convert a mathematician's written work into JSON. Copy the "
        "content faithfully: do not add claims, do not evaluate the "
        "mathematics, do not summarise beyond what is asked. "
        f"{instruction}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": source_text},
    ]

    # Schema-constrained first; drop to prompt-only if the grammar and the
    # model's default thinking mode conflict (symptom: empty content, output
    # stranded in message.thinking). The repair pipeline in clean_json_text
    # reclaims JSON from free-form output on the fallback path. The conflict
    # is a property of the model + Ollama version, not of one call, so once
    # discovered it's remembered for the rest of the run.
    global _SCHEMA_MODE_BROKEN
    use_schema = not _SCHEMA_MODE_BROKEN

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        payload: Dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": dict(EXTRACT_OPTIONS),
        }
        if use_schema:
            payload["format"] = schema
        else:
            # Free-form mode: the model may think first, so it needs a real
            # budget, and the schema moves from grammar to prompt.
            payload["options"]["num_predict"] = max(
                NUM_PREDICT_EXTRACT * 4, 4096
            )
            payload["messages"] = [
                {
                    "role": "system",
                    "content": system
                    + " Respond with ONLY a JSON object matching this schema, "
                    "no fences, no commentary: "
                    + json.dumps(schema),
                },
            ] + messages[1:]

        content: Optional[str] = None
        try:
            body = _post(payload)
            msg = body.get("message", {})
            content = msg.get("content", "") or ""
            if body.get("done_reason") == "length":
                log(f"  ⚠️  {role} extraction hit the token ceiling.", verbose)

            if use_schema and not content.strip():
                # Grammar/thinking conflict: qwen3.x thinks by default, the
                # format grammar suppresses the answer, and the output lands
                # in message.thinking. Deterministic, so don't retry same-mode.
                stranded = len(msg.get("thinking", "") or "")
                log(
                    f"  ⚠️  {role} extraction returned empty content with format "
                    f"set ({stranded} chars stranded in thinking). Falling back "
                    f"to prompt-only JSON for the rest of the run.",
                    verbose,
                )
                use_schema = False
                _SCHEMA_MODE_BROKEN = True
                continue

            return json.loads(clean_json_text(content))
        except (requests.RequestException, KeyError, TypeError) as e:
            log(f"  ⚠️  {role} extraction transport error (attempt {attempt}): {e}", verbose)
        except json.JSONDecodeError as e:
            log(f"  ⚠️  {role} extraction parse error (attempt {attempt}): {e}", verbose)
            if content is not None:
                log(f"     raw ({len(content)} chars) head: {content[:200]!r}", verbose)
            if content is not None and content.strip():
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid JSON ({e}). Reply with ONLY the "
                            "JSON object, no fences and no commentary."
                        ),
                    },
                ]

    log(f"  ❌ {role} extraction failed after all retries.", verbose)
    return {}


# ----------------------------------------------------------------------------
# Context filtering
# ----------------------------------------------------------------------------
def planner_dag_view(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Planner sees lemma statements + dependency structure, never proofs."""
    return {
        "proved_lemmas": {
            lid: {
                "statement": node["statement"],
                "dependencies": node.get("dependencies", []),
            }
            for lid, node in dag["lemmas"].items()
        }
    }


def dependency_context(
    dag: Dict[str, Any], dep_ids: List[str], include_proofs: bool = True
) -> Dict[str, Any]:
    """Context for the prover and verifiers.

    Prompt tokens and generation tokens share num_ctx, so every token spent
    here is a token the prover can't spend writing. The prover gets full
    proofs of its direct dependencies (it may need to see how they were
    established); the verifiers get statements only, since they are checking
    the new proof, not re-auditing accepted ones.
    """
    ctx: Dict[str, Any] = {
        "dependency_lemmas": {
            lid: (
                {
                    "statement": dag["lemmas"][lid]["statement"],
                    "proof": dag["lemmas"][lid]["proof"],
                }
                if include_proofs
                else {"statement": dag["lemmas"][lid]["statement"]}
            )
            for lid in dep_ids
            if lid in dag["lemmas"]
        }
    }
    if include_proofs:
        ctx["all_proved_statements"] = {
            lid: node["statement"] for lid, node in dag["lemmas"].items()
        }
    return ctx


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_loop(verbose: bool = True) -> Dict[str, Any]:
    """Execute the multi-agent theorem proving loop.

    Args:
        verbose: If True, print progress to the console.

    Returns:
        The final DAG, so callers running with --no-verbose still get a result.
    """
    conjecture = load_file(CONJECTURE_FILE)
    if not conjecture:
        log(f"{CONJECTURE_FILE} is empty — write your conjecture there first.", verbose)
        return load_dag()

    planner_sys = load_file("planner.md")
    prover_sys = load_file("prover.md")
    verifier_a_sys = load_file("verifier_a.md")
    verifier_b_sys = load_file("verifier_b.md")

    failed_attempts: Dict[str, List[str]] = {}

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---", verbose)
        dag = load_dag()

        # ---------------- Step 1: Planner ----------------
        planner_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}\n\n"
            f"Previously rejected attempts (avoid or decompose these):\n"
            f"{json.dumps(failed_attempts, indent=2)}"
        )
        plan_text, plan_status = reason(
            planner_sys, planner_user, "planner", THINK["planner"], verbose
        )
        if plan_status == "ceiling" and not plan_text:
            log("Planner exhausted its token budget. Re-planning.", verbose)
            continue
        if not plan_text:
            log("Planner produced nothing. Re-planning next iteration.", verbose)
            continue

        # planner.md demands raw JSON output; extraction is only the fallback.
        planner_res = parse_json_or_none(plan_text)
        if planner_res is None:
            planner_res = extract(
                "Extract the plan. is_conjecture_proved must be true only if the "
                "text explicitly concludes the conjecture is fully proved.",
                plan_text,
                PLANNER_SCHEMA,
                "planner",
                verbose,
            )

        if planner_res.get("is_conjecture_proved"):
            log("\n🎉 Conjecture has been fully proved!", verbose)
            dag["conjecture"] = conjecture
            dag["status"] = "proved"
            save_dag(dag)
            return dag

        next_lemma = planner_res.get("next_lemma") or {}
        lemma_id = next_lemma.get("id")
        lemma_stmt = next_lemma.get("statement")
        dep_ids = next_lemma.get("dependencies") or []

        if not lemma_id or not lemma_stmt:
            log("Planner failed to produce a valid next lemma. Re-planning.", verbose)
            continue
        if lemma_id in dag["lemmas"]:
            log(f"Planner re-proposed already-proved lemma {lemma_id}; skipping.", verbose)
            continue

        missing = [d for d in dep_ids if d not in dag["lemmas"]]
        if missing:
            log(f"Planner listed unproved dependencies {missing}; re-planning.", verbose)
            failed_attempts.setdefault(lemma_id, []).append(
                f"Planning error: depends on unproved lemmas {missing}."
            )
            continue

        log(f"📌 Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)

        # ---------------- Step 2: Prover ----------------
        # prover.md instructs the model to reply with {"lemma_id", "proof"}.
        # We parse that JSON locally rather than via a second model call, so
        # the proof text can never be abridged or paraphrased; if the model
        # ignored the format, its content is taken verbatim as the proof.
        deps_ctx = dependency_context(dag, dep_ids)
        prover_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n{json.dumps(deps_ctx, indent=2)}\n\n"
            f"Lemma to prove:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Feedback from previously rejected proofs of this lemma:\n"
            f"{json.dumps(failed_attempts.get(lemma_id, []), indent=2)}"
        )
        proof_text, prover_status = reason(
            prover_sys, prover_user, "prover", THINK["prover"], verbose
        )

        if prover_status == "ceiling" and not proof_text:
            log(
                f"⛔ Lemma {lemma_id} is too large to prove in one call. "
                f"Asking the planner to decompose it.",
                verbose,
            )
            failed_attempts.setdefault(lemma_id, []).append(
                "Lemma too large: the prover exhausted its entire token budget "
                "without completing a proof. Decompose this into smaller, "
                "independently provable lemmas rather than re-proposing it."
            )
            continue

        proof = proof_text
        prover_res = parse_json_or_none(proof_text)
        if prover_res is not None:
            candidate = prover_res.get("proof")
            if isinstance(candidate, str) and candidate.strip():
                proof = candidate.strip()

        if not proof:
            log(f"❌ Prover produced no proof for {lemma_id}.", verbose)
            failed_attempts.setdefault(lemma_id, []).append("Prover returned nothing.")
            continue

        log(f"✍️ Proof generated for {lemma_id} ({len(proof)} chars).", verbose)

        # ---------------- Steps 3 & 4: Independent verifiers ----------------
        verifier_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n"
            f"{json.dumps(dependency_context(dag, dep_ids, include_proofs=False), indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
        )

        verdicts: Dict[str, Dict[str, str]] = {}
        for tag, sys_prompt in (("A", verifier_a_sys), ("B", verifier_b_sys)):
            review, review_status = reason(
                sys_prompt, verifier_user, "verifier", THINK["verifier"], verbose
            )
            if review_status == "ceiling" and not review:
                verdicts[tag] = {
                    "decision": "reject",
                    "justification": (
                        "Verifier exhausted its token budget without reaching a "
                        "verdict; the proof is likely too long to review in one pass."
                    ),
                }
                continue
            # verifier_*.md demand raw JSON output; extraction is the fallback.
            res = parse_json_or_none(review) if review else None
            if res is None and review:
                res = extract(
                    "Extract the verdict. decision is 'accept' only if the review "
                    "endorses the proof without unresolved objections.",
                    review,
                    VERIFIER_SCHEMA,
                    f"verifier {tag}",
                    verbose,
                )
            res = res or {}
            verdicts[tag] = {
                "decision": str(res.get("decision", "")).strip().lower(),
                "justification": res.get("justification", "(no justification)"),
            }

        labels = {"A": "logic", "B": "computation"}
        for tag in ("A", "B"):
            decision = verdicts[tag]["decision"].upper() or "???"
            log(
                f"🔍 Verifier {tag} ({labels[tag]}): {decision} — "
                f"{verdicts[tag]['justification']}",
                verbose,
            )

        # ---------------- Step 5: Consensus & DAG update ----------------
        if all(verdicts[t]["decision"] == "accept" for t in ("A", "B")):
            log(f"✅ Lemma {lemma_id} accepted by both verifiers! Adding to DAG.", verbose)
            dag["lemmas"][lemma_id] = {
                "statement": lemma_stmt,
                "proof": proof,
                "dependencies": dep_ids,
            }
            save_dag(dag)
            failed_attempts.pop(lemma_id, None)
        else:
            log(f"❌ Proof rejected for {lemma_id}. Recording feedback for retry.", verbose)
            for tag in ("A", "B"):
                if verdicts[tag]["decision"] != "accept":
                    failed_attempts.setdefault(lemma_id, []).append(
                        f"Verifier {tag}: {verdicts[tag]['justification']}"
                    )

    log(
        f"\n⏹ Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without completing the proof.",
        verbose,
    )
    return load_dag()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    global MODEL_NAME, CONJECTURE_FILE, DAG_FILE, MAX_ITERATIONS, NUM_CTX

    parser = argparse.ArgumentParser(
        description="Run the multi-agent theorem prover."
    )
    # Adds --verbose / --no-verbose flags (defaults to True)
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable console logging (default: --verbose)",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model tag")
    parser.add_argument("--conjecture", default=CONJECTURE_FILE, help="Conjecture file")
    parser.add_argument("--dag", default=DAG_FILE, help="DAG state file")
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Loop iterations before giving up (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=NUM_CTX,
        help=f"Context window in tokens (default: {NUM_CTX}). Lower it if VRAM is tight.",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model
    CONJECTURE_FILE = args.conjecture
    DAG_FILE = args.dag
    MAX_ITERATIONS = args.max_iterations
    NUM_CTX = args.num_ctx
    REASONING_OPTIONS["num_ctx"] = NUM_CTX
    EXTRACT_OPTIONS["num_ctx"] = NUM_CTX

    run_loop(verbose=args.verbose)


if __name__ == "__main__":
    main()
