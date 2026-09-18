# Multi-Agent Theorem Prover

A Python program that drives a small team of LLM agents to iteratively prove
(or refute) a mathematical conjecture. The agents work in a loop —
**planner → prover → verifier → reviser** — and accumulate a proof as a DAG of
lemmas written to `dag.json` next to the conjecture.

## How it works

Each iteration:

1. **Planner** (`planner.md`) reads the conjecture, the lemmas proved so far,
   and any previously rejected attempts, and proposes
   `PLANNER_CANDIDATES` (5) candidate lemmas, best-first. Each candidate
   carries an *aim* — `proof` (work toward the conjecture being true) or
   `counterexample` (work toward refuting it).
2. **Selection** picks exactly one candidate:
   - `--mode auto`: `screen_candidates()` filters, and the first survivor is
     taken.
   - `--mode human`: a menu appears. Type a number and Enter to accept that
     lemma as true on your own authority (it goes straight into the DAG,
     marked `"provenance": "operator"`). Type `Np` to send a lemma through
     the ordinary proving pipeline instead. You can also write your own
     lemma (`w`, or `wp` to have it proved) or send the planner back.
3. **Prover** (`prover.md`) writes a complete, rigorous proof of the chosen
   lemma, citing only lemmas already in the DAG.
4. **Verifier** (`verifier.md`) checks it adversarially — logic *and*
   arithmetic. A lemma enters the DAG only after `VERIFY_PASSES` (default 3)
   independent passes have all accepted it.
5. **Reviser** (`reviser.md`) handles a reject: it decides whether the fault
   is in the *proof* (send the prover back with the verdict as feedback) or
   in the *statement* (return a revised statement as the prover's new target).
   After `MAX_PROOF_ATTEMPTS` (default 12) prover rounds the lemma is given
   up and the planner is asked again.

### Thinking vs. structured output

The prompts instruct each model to emit strict JSON directly, so every
response is first parsed as-is; a schema-constrained *extraction* call only
runs as a fallback. The underlying tension between `format` (JSON schemas)
and `think` (reasoning) differs between the two supported backends and is
handled inside `src/llm_backend.py`, which `src/main.py` uses without knowing
which server is listening.

### Backends

Two transports, probed at startup (`src/llm_backend.py`):

- **Ollama** — native `/api/chat`; `think` as a top-level field, `format` as
  a JSON schema, reasoning returned in `message.thinking`.
- **llama.cpp** — `server` HTTP API; there the grammar suppresses thinking,
  so reasoning is disabled for schema calls.

Model capabilities (context size, thinking support, thinking levels) are
*profiles*, probed rather than hard-coded, so adding a model needs no code
change.

## Setup

A Nix dev shell is provided:

```sh
nix develop
```

It gives you a Python 3 environment with `requests` installed. Otherwise any
Python 3 install with `requests` works.

## Usage

```sh
python src/main.py --conjecture torsor_examples
python src/main.py --conjecture torsor_examples --model gemma4:31b
python src/main.py --conjecture torsor_examples --backend llamacpp \
               --model Qwen3.5-122B-Q4_K_M
python src/main.py --conjecture torsor_examples --no-verbose --max-iterations 25
python src/main.py --conjecture torsor_examples --mode human
python src/main.py --conjecture torsor_examples --verify-passes 5
```

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--conjecture NAME` | `curve_indices` | A directory under `conjectures/` holding a `conjecture.md`. |
| `--backend {ollama,llamacpp}` | `ollama` | Which server to talk to. |
| `--model NAME` | `qwen3.6:35b` | Ollama model tag, or the label llama-server reports. |
| `--host URL` | per backend | Server base URL (`localhost:11434` for ollama, `localhost:8081` for llamacpp). |
| `--dag PATH` | `conjectures/<name>/dag.json` | Use a separate DAG file (e.g. to isolate a model's run). |
| `--max-iterations N` | 10 | Loop iterations before giving up. |
| `--verify-passes N` | 3 | Independent verifier passes a proof must survive. |
| `--max-proof-attempts N` | 12 | Prover rounds per lemma before it is given up. |
| `--num-ctx N` | 40960 | Context window in tokens (capped at the server's limit; lower it if VRAM is tight). |
| `--mode {auto,human}` | `auto` | Automated selection vs. human menu at the planning step. |
| `--hotkey KEY` | `h` | Keystroke that toggles auto/human mid-run (pass `''` to disable). |
| `--verbose / --no-verbose` | on | Console logging. |

### Modes and the hotkey

`--mode auto` runs unattended. `--mode human` puts you in the driver's seat
at the planning step. You can cross between them mid-run, in either
direction, without restarting:

- Press the hotkey (`h`) during any agent call and the mode flips at the
  next planning step.
- At the menu itself, type `a` (the line-oriented input is what the menu
  already owns, so the hotkey is paused while it is up).

`src/interaction.py` explains why the toggle is a keystroke in one place and
a line of input in the other.

## Project layout

```
agents/
  planner.md  prover.md  verifier.md  reviser.md   # default agent prompts
src/
  main.py              # the proof loop (planner → prover → verifier → reviser)
  llm_backend.py       # backend transports (Ollama, llama.cpp) + model profiles
  interaction.py       # human-in-the-loop menu + hotkey
  workspace.py         # per-conjecture run directories and prompt resolution
conjectures/
  torsor_examples/
    conjecture.md          # required: the statement
    dag.json               # shared by every model and backend
    prover.md              # optional per-conjecture prompt override
flake.nix                  # Nix dev shell
```

### Conjectures and the shared DAG

`--conjecture` names a directory under `conjectures/` that must contain a
`conjecture.md`. The DAG is written beside it as `dag.json` and is *shared by
every model and backend*: point a second model at a conjecture already under
way and it continues from the lemmas the first one proved. A lemma proved by
a 27B model and passed by the verifiers is no less proved when a 122B model
arrives. The cost is that this is collaboration, not a controlled comparison —
give a model its own file with `--dag` if you want them isolated again.

Prompt files are looked up in the conjecture directory first, then in
`agents/`, so a conjecture that needs a special prompt (e.g. a prover primed
for Brauer groups) can carry one without forking the defaults.
