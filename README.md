# Multi-Agent Theorem Prover

A Python program that drives a small team of LLM agents to iteratively prove
(or refute) a mathematical conjecture. The agents work in a loop —
**planner → selector → prover → verifier → reviser** — and accumulate a proof
as a DAG of lemmas written to `dag.json` next to the conjecture. (In
`--mode human`, the operator takes the selector's place at the menu.)

## How it works

Each iteration:

1. **Planner** (`planner.md`) reads the conjecture, the lemmas proved so far,
   and any previously rejected attempts, and proposes
   `PLANNER_CANDIDATES` (5) candidate lemmas, best-first. Each candidate
   carries an *aim* — `proof` (work toward the conjecture being true) or
   `counterexample` (work toward refuting it).
2. **Selection** picks exactly one candidate:
   - `--mode auto`: `screen_candidates()` filters out candidates whose ids are
     already in the DAG, and when more than one survives, the **selector**
     (`selector.md`) weighs them all and picks the one it judges most likely
     to come through the prover and the verifiers. The planner's best-first
     order is the fallback: it is taken when the selector fails or when only
     one candidate survives.
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

### When a call hits the context wall

The server is stateless, so a response cut off by the context window cannot
just be re-sent. Before reporting a ceiling, `reason()` runs pi-style
compaction: the thinking trace (scratch) is summarised into a structured
summary — in chunked passes if it is big — the answer written so far is kept
verbatim, and the model is asked to resume from the cut with the whole
headroom of the now-smaller prompt. The passes are unbounded: a continuation
that hits the wall itself becomes the next pass's input, its trace folded
into the summary, its text extending the verbatim answer, until the answer
completes. Only when a pass leaves nothing to build on (or no headroom left)
is the lemma reported as too large for the window, and the usual
decomposition escape hatches take over.

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
python src/main.py --conjecture algebra_example
python src/main.py --conjecture algebra_example --model gemma4:31b
python src/main.py --conjecture algebra_example --backend llamacpp \
               --model Qwen3.5-122B-Q4_K_M
python src/main.py --conjecture algebra_example --no-verbose --max-iterations 25
python src/main.py --conjecture algebra_example --mode human
python src/main.py --conjecture algebra_example --verify-passes 5

# once, before proving: parse the conjecture's reference collection
python src/parsing.py --conjecture algebra_example
```

`parsing.py` shares main.py's backend options (`--backend`, `--host`,
`--model`, `--num-ctx`) and its `--conjecture NAME` / `--verbose` pair; it
has no proving options, because it has no loop.

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--conjecture NAME` | `algebra_example` | A directory under `conjectures/` holding a `conjecture.md`. |
| `--backend {ollama,llamacpp}` | `ollama` | Which server to talk to. |
| `--model NAME` | `qwen3.6:35b` | Ollama model tag, or the label llama-server reports. |
| `--host URL` | per backend | Server base URL (`localhost:11434` for ollama, `localhost:8081` for llamacpp). |
| `--dag PATH` | `conjectures/<name>/dag.json` | Use a separate DAG file (e.g. to isolate a model's run). |
| `--max-iterations N` | 10 | Loop iterations before giving up. |
| `--verify-passes N` | 3 | Independent verifier passes a proof must survive. |
| `--max-proof-attempts N` | 12 | Prover rounds per lemma before it is given up. |
| `--num-ctx N` | 65536 | Context window in tokens (capped at the server's limit; lower it if VRAM is tight). |
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
  planner.md  selector.md  prover.md  verifier.md  reviser.md   # default agent prompts
  parsing.md                               # references parser prompt
src/
  main.py              # the proof loop (planner → selector → prover → verifier → reviser)
  parsing.py           # standalone references/ → references.md parser
  llm_backend.py       # backend transports (Ollama, llama.cpp) + model profiles
  interaction.py       # human-in-the-loop menu + hotkey
  workspace.py         # per-conjecture run directories and prompt resolution
conjectures/
  algebra_example/
    conjecture.md          # required: the statement
    dag.json               # shared by every model and backend
    references/            # optional: source files (.tex/.md/plain text) to parse
    references.md          # parsed theorem-level results (strict JSON array)
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

### References

A conjecture can ship a `references/` directory: chapters, surveys or
papers, as `.tex`, `.md` or plain text, that the agents may lean on. The
directory is parsed by a standalone tool, not by the proving loop:

```sh
python src/parsing.py --conjecture algebra_example
```

reads every parsable file in `references/`, asks the model (via
`agents/parsing.md`) for the theorem-level results, and writes them to
`references.md` beside the DAG as a **strict JSON array** — one object per
result with `id`, `slogan`, `formal statement`, `reference` (where it appears
in the file) and `tags`. `main.py` then feeds that file to the loop at three
different granularities:

- **Planner** — an `id` + `slogan` shortlist, and only while the collection
  is small: below `PLANNER_REFERENCE_LIMIT` (20) entries it gets every
  result, at or above it none. A shortlist of hundreds of slogans costs
  more than it returns, and the prover sees the full collection anyway, so
  a named result the planner misses is one round away, not lost. A strategy
  for large collections is deliberately not built yet.
- **Prover** — the whole collection unconditionally (id, slogan, formal
  statement, tags). It may cite any entry by `id` in `cited_lemmas` and
  rely on it without proving it, exactly like a proved lemma.
- **Verifier** — only the formal statements of the results the proof
  actually cites, merged with the cited DAG lemmas into one set. The rest
  of the collection is withheld on purpose: a proof that leans on a result
  it never cited must read as an unjustified leap, and the verifier is
  told that citing a reference is legitimate while proving it inline is
  not.

Two properties of the parser matter for resuming:

- **Ids are stable across re-parses.** `ref_N` numbers are assigned by the
  tool, never by the model; a result whose formal statement matches an
  existing entry keeps its id, so a DAG that cites `ref_3` still means the
  same thing after you add a chapter to `references/` and re-run.
- **Re-parses are a union.** Entries from a previous `references.md` whose
  statements were not re-derived this run are kept, so a flaky parse of one
  file cannot silently delete a result another proof cites. An empty
  `references/` directory never touches an existing `references.md`. There is
  no clear mode: to reset the collection, delete or rewrite `references.md`
  (a DAG that cited its entries will warn about the dangling ids) — deleting
  it and re-running `parsing.py` is the cleanest way to start over.

A run without `references/` or `references.md` is exactly the old run:
every prompt degrades to its pre-references form, which is also the failure
mode for a corrupt `references.md` (it warns once and is ignored).
