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

The prover skips extraction altogether: its only payload is the proof text, so
we take the reasoning stage's content verbatim rather than round-tripping it
through a second model call that might abridge it.

Usage
-----
    python main.py                       # verbose, defaults
    python main.py --no-verbose          # silent
    python main.py --max-iterations 25 --no-traces
    python main.py --model qwen3.6:27b --conjecture other.md
"""

import argparse
import json
import os
import re
from datetime import datetime
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
TRACE_DIR = "traces"         # Chain-of-thought transcripts land here
MAX_ITERATIONS = 10
LLM_MAX_RETRIES = 3
REQUEST_TIMEOUT = 1800       # Thinking models are slow; give them room

# Ollama defaults num_ctx low (2048/4096) regardless of model capability.
# Reasoning tokens are drawn from the same budget as the answer, so a thinking
# prover needs a much larger allowance than a one-shot one.
NUM_CTX = 40960              # model supports 256K; raise if VRAM allows
NUM_PREDICT_REASONING = 12288  # ~3B active params => generation is cheap
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
    "num_predict": NUM_PREDICT_REASONING,
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

DEBUG_RAW = True     # Dump raw output when JSON parsing fails
SAVE_TRACES = True   # Write per-call reasoning transcripts to TRACE_DIR


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


def save_trace(iteration: int, role: str, thinking: str, output: str) -> None:
    """Persist a call's chain of thought. Invaluable for working out *why* a
    verifier rejected something, or where a proof went off the rails."""
    if not SAVE_TRACES:
        return
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"iter{iteration:02d}_{role}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {role} — iteration {iteration}\n")
        f.write(f"_{datetime.now().isoformat(timespec='seconds')}_\n\n")
        if thinking:
            f.write(f"## Reasoning\n\n{thinking}\n\n")
        f.write(f"## Output\n\n{output}\n")


# ----------------------------------------------------------------------------
# JSON repair (fallback path; the schema grammar should make it unnecessary)
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_BAD_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
# \frac, \times, \to, \nabla begin with legal JSON escape chars, so a naive
# repair silently turns them into formfeeds and tabs. Trailing lowercase means
# it was a LaTeX command, not a control character.
_LATEX_ESCAPE_RE = re.compile(r"\\([bfnrtu][a-z]+)")


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
    """Normalise model output into parseable JSON: strip fences, inline think
    blocks and prose; repair LaTeX escapes; salvage truncation."""
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text).strip()
    text = _extract_json_object(text)
    text = _BAD_ESCAPE_RE.sub(r"\\\\", text)
    text = _LATEX_ESCAPE_RE.sub(r"\\\\\1", text)
    try:
        json.loads(text)
    except json.JSONDecodeError:
        text = _salvage_truncated(text)
    return text


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def reason(
    system_prompt: str,
    user_prompt: str,
    role: str,
    think: Any = True,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, thinking). Ollama puts the chain of thought in
    `message.thinking` when the model supports separated thinking; models that
    inline <think> tags are handled by the fallback strip.
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
        },
    }
    # Only send `think` when we actually want it on. Never send False.
    if think is not None and think is not False:
        payload["think"] = think

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            body = _post(payload)
            msg = body.get("message", {})
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""

            if body.get("done_reason") == "length":
                log(
                    f"  ⚠️  {role} hit the token ceiling "
                    f"(num_predict={NUM_PREDICT_REASONING}). Reasoning may be cut off.",
                    verbose,
                )
            if not thinking and "<think>" in content:
                # Model inlined its reasoning instead of separating it.
                m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if m:
                    thinking = m.group(1).strip()
                content = _THINK_RE.sub("", content).strip()
            if content.strip():
                return content.strip(), thinking.strip()
            log(f"  ⚠️  {role} returned empty content (attempt {attempt}).", verbose)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} transport error (attempt {attempt}): {e}", verbose)

    return "", ""


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

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": dict(EXTRACT_OPTIONS),
        }
        content: Optional[str] = None
        try:
            body = _post(payload)
            content = body.get("message", {}).get("content", "")
            if body.get("done_reason") == "length":
                log(f"  ⚠️  {role} extraction hit the token ceiling.", verbose)
            return json.loads(clean_json_text(content))
        except (requests.RequestException, KeyError, TypeError) as e:
            log(f"  ⚠️  {role} extraction transport error (attempt {attempt}): {e}", verbose)
        except json.JSONDecodeError as e:
            log(f"  ⚠️  {role} extraction parse error (attempt {attempt}): {e}", verbose)
            if DEBUG_RAW and content is not None:
                log(f"     raw ({len(content)} chars) head: {content[:200]!r}", verbose)
                log(f"     raw tail: {content[-200:]!r}", verbose)
            if content is not None:
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


def dependency_context(dag: Dict[str, Any], dep_ids: List[str]) -> Dict[str, Any]:
    """Prover/verifiers see full proofs only for direct dependencies."""
    return {
        "dependency_lemmas": {
            lid: {
                "statement": dag["lemmas"][lid]["statement"],
                "proof": dag["lemmas"][lid]["proof"],
            }
            for lid in dep_ids
            if lid in dag["lemmas"]
        },
        "all_proved_statements": {
            lid: node["statement"] for lid, node in dag["lemmas"].items()
        },
    }


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
        plan_text, plan_think = reason(
            planner_sys, planner_user, "planner", THINK["planner"], verbose
        )
        save_trace(iteration, "planner", plan_think, plan_text)
        if not plan_text:
            log("Planner produced nothing. Re-planning next iteration.", verbose)
            continue

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
        # No extraction stage: the proof text *is* the payload, and a second
        # pass could only abridge or paraphrase it.
        deps_ctx = dependency_context(dag, dep_ids)
        prover_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n{json.dumps(deps_ctx, indent=2)}\n\n"
            f"Lemma to prove:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Feedback from previously rejected proofs of this lemma:\n"
            f"{json.dumps(failed_attempts.get(lemma_id, []), indent=2)}"
        )
        proof, proof_think = reason(
            prover_sys, prover_user, "prover", THINK["prover"], verbose
        )
        save_trace(iteration, f"prover_{lemma_id}", proof_think, proof)

        if not proof:
            log(f"❌ Prover produced no proof for {lemma_id}.", verbose)
            failed_attempts.setdefault(lemma_id, []).append("Prover returned nothing.")
            continue

        log(f"✍️ Proof generated for {lemma_id} ({len(proof)} chars).", verbose)

        # ---------------- Steps 3 & 4: Independent verifiers ----------------
        verifier_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n{json.dumps(deps_ctx, indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
        )

        verdicts: Dict[str, Dict[str, str]] = {}
        for tag, sys_prompt in (("A", verifier_a_sys), ("B", verifier_b_sys)):
            review, review_think = reason(
                sys_prompt, verifier_user, "verifier", THINK["verifier"], verbose
            )
            save_trace(iteration, f"verifier{tag}_{lemma_id}", review_think, review)
            res = (
                extract(
                    "Extract the verdict. decision is 'accept' only if the review "
                    "endorses the proof without unresolved objections.",
                    review,
                    VERIFIER_SCHEMA,
                    f"verifier {tag}",
                    verbose,
                )
                if review
                else {}
            )
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
    global MODEL_NAME, CONJECTURE_FILE, DAG_FILE, TRACE_DIR
    global MAX_ITERATIONS, SAVE_TRACES, NUM_CTX

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
    parser.add_argument(
        "--traces",
        action=argparse.BooleanOptionalAction,
        default=SAVE_TRACES,
        help="Write chain-of-thought transcripts to the trace directory "
             "(default: --traces). Independent of --verbose.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model tag")
    parser.add_argument("--conjecture", default=CONJECTURE_FILE, help="Conjecture file")
    parser.add_argument("--dag", default=DAG_FILE, help="DAG state file")
    parser.add_argument("--trace-dir", default=TRACE_DIR, help="Trace output directory")
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
    TRACE_DIR = args.trace_dir
    MAX_ITERATIONS = args.max_iterations
    SAVE_TRACES = args.traces
    NUM_CTX = args.num_ctx
    REASONING_OPTIONS["num_ctx"] = NUM_CTX
    EXTRACT_OPTIONS["num_ctx"] = NUM_CTX

    run_loop(verbose=args.verbose)


if __name__ == "__main__":
    main()
