import json
import os
import re
import argparse
import requests
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:32b"  # Replace with your local model tag
DAG_FILE = "dag.json"
MAX_ITERATIONS = 10
LLM_MAX_RETRIES = 3          # Retries *inside* a single call_llm
REQUEST_TIMEOUT = 600        # Seconds; local 32B models can be slow

# Per-role sampling temperatures.
TEMPERATURES = {
    "planner": 0.0,
    "prover": 0.0,
    "verifier": 0.3,
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
    """Load current DAG state from file."""
    if os.path.exists(DAG_FILE):
        with open(DAG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lemmas": {}}


def save_dag(dag: Dict[str, Any]) -> None:
    """Save updated DAG state to file."""
    with open(DAG_FILE, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2)


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def clean_json_text(raw: str) -> str:
    """Strip markdown code fences and grab the outermost JSON object."""
    text = _FENCE_RE.sub("", raw).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = LLM_MAX_RETRIES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Call the local LLM and enforce a JSON return object."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return json.loads(clean_json_text(content))
        except (requests.RequestException, KeyError) as e:
            if verbose:
                print(f"  ⚠️  LLM transport error (attempt {attempt}/{max_retries}): {e}")
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ⚠️  JSON parse error (attempt {attempt}/{max_retries}): {e}")
            messages = messages[:2] + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON "
                        f"({e}). Respond again with ONLY the valid JSON object, "
                        "no markdown fences, no commentary."
                    ),
                },
            ]

    if verbose:
        print("  ❌ LLM call failed after all retries.")
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
    """Prover/verifiers see statements + proofs of direct dependencies."""
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
def run_loop(verbose: bool = True) -> None:
    """Execute the multi-agent theorem proving loop.
    
    Args:
        verbose (bool): If True, output progress prints to standard console.
    """
    def log(message: str) -> None:
        if verbose:
            print(message)

    conjecture = load_file("conjecture.md")
    if not conjecture:
        log("conjecture.md is empty — write your conjecture there first.")
        return

    planner_sys = load_file("planner.md")
    prover_sys = load_file("prover.md")
    verifier_a_sys = load_file("verifier_a.md")
    verifier_b_sys = load_file("verifier_b.md")

    failed_attempts: Dict[str, List[str]] = {}

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
        dag = load_dag()

        # ---------------- Step 1: Planner ----------------
        planner_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}\n\n"
            f"Previously rejected attempts (avoid or decompose these):\n"
            f"{json.dumps(failed_attempts, indent=2)}"
        )
        planner_res = call_llm(planner_sys, planner_user, TEMPERATURES["planner"], verbose=verbose)

        if planner_res.get("is_conjecture_proved"):
            log("\n🎉 Conjecture has been fully proved!")
            dag["conjecture"] = conjecture
            dag["status"] = "proved"
            save_dag(dag)
            return

        next_lemma = planner_res.get("next_lemma") or {}
        lemma_id = next_lemma.get("id")
        lemma_stmt = next_lemma.get("statement")
        dep_ids = next_lemma.get("dependencies", []) or []

        if not lemma_id or not lemma_stmt:
            log("Planner failed to produce a valid next lemma. Re-planning next iteration.")
            continue

        if lemma_id in dag["lemmas"]:
            log(f"Planner re-proposed already-proved lemma {lemma_id}; skipping.")
            continue

        missing = [d for d in dep_ids if d not in dag["lemmas"]]
        if missing:
            log(f"Planner listed unproved dependencies {missing}; re-planning.")
            failed_attempts.setdefault(lemma_id, []).append(
                f"Planning error: depends on unproved lemmas {missing}."
            )
            continue

        log(f"📌 Next Lemma [{lemma_id}]: {lemma_stmt}")

        # ---------------- Step 2: Prover ----------------
        deps_ctx = dependency_context(dag, dep_ids)
        prior_feedback = failed_attempts.get(lemma_id, [])
        prover_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n{json.dumps(deps_ctx, indent=2)}\n\n"
            f"Lemma to prove:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Feedback from previously rejected proofs of this lemma:\n"
            f"{json.dumps(prior_feedback, indent=2)}"
        )
        prover_res = call_llm(prover_sys, prover_user, TEMPERATURES["prover"], verbose=verbose)
        proof = prover_res.get("proof", "")

        if not proof:
            log(f"❌ Prover produced no proof for {lemma_id}.")
            failed_attempts.setdefault(lemma_id, []).append("Prover returned an empty proof.")
            continue

        log(f"✍️ Proof generated for {lemma_id}.")

        # ---------------- Steps 3 & 4: Independent verifiers ----------------
        verifier_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n{json.dumps(deps_ctx, indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(next_lemma, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
        )
        v1 = call_llm(verifier_a_sys, verifier_user, TEMPERATURES["verifier"], verbose=verbose)
        v2 = call_llm(verifier_b_sys, verifier_user, TEMPERATURES["verifier"], verbose=verbose)

        v1_dec = str(v1.get("decision", "")).strip().lower()
        v2_dec = str(v2.get("decision", "")).strip().lower()
        v1_just = v1.get("justification", "(no justification)")
        v2_just = v2.get("justification", "(no justification)")

        log(f"🔍 Verifier A (logic): {v1_dec.upper() or '???'} — {v1_just}")
        log(f"🔍 Verifier B (computation): {v2_dec.upper() or '???'} — {v2_just}")

        # ---------------- Step 5: Consensus & DAG update ----------------
        if v1_dec == "accept" and v2_dec == "accept":
            log(f"✅ Lemma {lemma_id} accepted by both verifiers! Adding to DAG.")
            dag["lemmas"][lemma_id] = {
                "statement": lemma_stmt,
                "proof": proof,
                "dependencies": dep_ids,
            }
            save_dag(dag)
            failed_attempts.pop(lemma_id, None)
        else:
            log(f"❌ Proof rejected for {lemma_id}. Recording feedback for retry.")
            reasons = []
            if v1_dec != "accept":
                reasons.append(f"Verifier A: {v1_just}")
            if v2_dec != "accept":
                reasons.append(f"Verifier B: {v2_just}")
            failed_attempts.setdefault(lemma_id, []).extend(reasons)

    log(f"\n⏹ Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without completing the proof.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multi-agent theorem prover."
    )

    # Adds --verbose / --no-verbose flags (Defaults to True)
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable console logging (default: --verbose)",
    )

    args = parser.parse_args()
    run_loop(verbose=args.verbose)
