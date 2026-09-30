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
4. **Verifiers** (`verifier_1.md`, `verifier_2.md`, `verifier_3.md`) check it
   adversarially — logic *and* arithmetic. Verification is three atomic
   steps, one per agent; a lemma enters the DAG only after all three have
   accepted it. (The three files are currently identical.)
5. **Reviser** (`reviser.md`) handles a reject, judging the lemma's
   difficulty from the rejected proof itself (the *last proof*). It decides
   whether the fault is in the *proof* (keep the statement; send the prover
   back with the verdict as feedback), in the *statement* (return a revised
   statement as the prover's new target), or whether the lemma is *too hard*
   to prove as stated (decompose it into a smaller lemma — a fresh id and a
   sub-statement the last proof left unjustified — and start the prover on
   that with a fresh budget). After `MAX_PROOF_ATTEMPTS` (default 3) prover
   rounds the lemma is given up and the planner is asked again. When the
   prover instead overflows the context window, the same job falls to
   `reviser_incomplete.md`, working from the partial proof — see
   [When a call hits the context wall](#when-a-call-hits-the-context-wall).

### Thinking vs. structured output

The prompts instruct each model to emit strict JSON directly, so every
response is first parsed as-is; a schema-constrained *extraction* call only
runs as a fallback. The underlying tension there — on llama.cpp, reasoning
silently voids the grammar — is handled inside `src/llm_backend.py`, which
disables reasoning for exactly the calls that carry a schema.

### When a call hits the context wall

The server is stateless, so a response cut off by the context window cannot
just be re-sent. Before reporting a ceiling, `reason()` runs a pi-style
compaction rescue: the thinking trace (scratch) is summarised into a
structured summary — in chunked passes if it is big — the answer written so
far is kept verbatim, and the model is asked to resume from the cut with the
whole headroom of the now-smaller prompt. A continuation that hits the wall
itself is not a failure but the next pass: its trace is folded into the
summary, its text extends the verbatim answer, and the model resumes from
the new cut.

The rescue is bounded to two compaction-and-resume passes
(`MAX_COMPACTION_PASSES`): the first usually finishes an answer that merely
ran long, the second covers the genuinely long proofs. Still unfinished then
— or when a pass leaves nothing to build on, or no headroom is left —
`reason()` reports the ceiling and hands back the partial work. The proof
loop compacts it a final time (folding the last cut-off trace into the
summary) and sends the summary plus the verbatim partial answer to
`reviser_incomplete.md`, a variant of the reviser that reads an overflow
instead of a rejected proof and picks a smaller lemma the partial work was
building toward. If it finds one, the overflowing lemma is set aside — its
history recorded under its own id — and the prover starts on the smaller
lemma with a fresh budget. Only when there is nothing left to build on, or
the reviser cannot pick a usable smaller lemma, does the lemma fall back to
the planner's decomposition escape hatches.

### Backend

One transport, probed at startup (`src/llm_backend.py`): the
OpenAI-compatible `/v1/chat/completions` endpoint. The same code serves two
backends:

- **llama.cpp** — a local `llama-server`. There the grammar suppresses
  thinking, so reasoning is disabled for schema calls.
- **a hosted API** — any OpenAI-compatible provider (OpenAI, OpenRouter,
  DeepSeek, ...), pointed at by `--host` with its base URL and authenticated
  with `--api-key` (or `$LLM_API_KEY`). A supplied key is checked against the
  provider's `/v1/models` at startup, so a wrong key fails loudly instead of
  401-ing every call in a run.

The probe can't tell the two apart by name, so it tries: a `/props` answer
means llama.cpp (full option set, real context ceiling); no `/props` means a
generic endpoint (standard OpenAI option set — `top_k`/`min_p` are dropped,
because a strict provider 400s on them — and the context limit is a default
unless `--num-ctx` is given).

`--backend {auto,llamacpp,openai}` (default `auto`) overrides that decision
when the probe can't see the real server — a proxy in front of llama-server
that eats `/props`, or a provider that happens to serve one. An explicit
backend is taken on faith and the banner warns when it contradicts the probe
(a forced llama.cpp against a server with no `/props` route, or a forced
standard OpenAI against one that has it). The probe's *measurements* — the
context ceiling and thinking support — are kept either way, because they
describe the server that is actually there.

Model capabilities (context size, thinking support) are *profiles*, probed
rather than hard-coded, so adding a model needs no code change.

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
python src/main.py --conjecture algebra_example --model Qwen3.5-122B-Q4_K_M
python src/main.py --conjecture algebra_example --no-verbose --max-iterations 25
python src/main.py --conjecture algebra_example --mode human

# or against a hosted API instead of a local llama-server
LLM_API_KEY=sk-... python src/main.py --conjecture algebra_example \
    --host https://api.openai.com/v1 --model gpt-4o

# once, before proving: parse the conjecture's reference collection
python src/parsing.py --conjecture algebra_example
```

`parsing.py` shares main.py's server options (`--host`, `--api-key`,
`--backend`, `--model`, `--num-ctx`) and its `--conjecture NAME` /
`--verbose` pair; it has no proving options, because it has no loop.

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--conjecture NAME` | `algebra_example` | A directory under `conjectures/` holding a `conjecture.md`. |
| `--model NAME` | `qwen3.8-27b` | Model name: the `--alias` llama-server was launched with, or a provider's model id (e.g. `gpt-4o`). |
| `--host URL` | `http://localhost:8081` | OpenAI-compatible base URL — a local llama-server, or a hosted API's base (e.g. `https://api.openai.com/v1`). `$LLAMA_HOST` overrides the default; 8080 is taken by open-webui. |
| `--api-key KEY` | `$LLM_API_KEY` | Sent as `Authorization: Bearer <key>`. Needed for a hosted API; leave unset for a plain local llama-server. |
| `--backend {auto,llamacpp,openai}` | `auto` | Which dialect of the OpenAI-compatible endpoint to speak: the full llama.cpp option set (with thinking suppressed for schema calls), or the standard OpenAI set. `auto` takes the probe's answer; the other two force it, for when the probe can't see the real server. |
| `--dag PATH` | `conjectures/<name>/dag.json` | Use a separate DAG file (e.g. to isolate a model's run). |
| `--max-iterations N` | 10 | Loop iterations before giving up. |
| `--max-proof-attempts N` | 3 | Prover rounds per lemma before it is given up. |
| `--fresh` | off | Ignore and delete the checkpoint (`<dag>.checkpoint.json`), restarting the iteration budget from 1. The DAG is kept. |
| `--num-ctx N` | the server's own (probed) | Context window in tokens. Defaults to the server's context — the `-c` llama-server was launched with, as probed. Lower it if VRAM is tight. |
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
- Ctrl-C stops the run from anywhere — the menu included — writing the
  checkpoint first; see [Cancelling a run](#cancelling-a-run-ctrl-c-and-resuming).
  At the menu, `q` does the same thing.

`src/interaction.py` explains why the toggle is a keystroke in one place and
a line of input in the other.

## Project layout

```
agents/
  planner.md  selector.md  prover.md  verifier_1/2/3.md  reviser.md   # default agent prompts
  reviser_incomplete.md                     # overflow reviser (partial proofs)
  parsing.md                               # references parser prompt
src/
  main.py              # the proof loop (planner → selector → prover → verifier → reviser)
  parsing.py           # standalone references/ → references.md parser
  llm_backend.py       # OpenAI-compatible transport (llama.cpp or a hosted API) + model profiles
  interaction.py       # human-in-the-loop menu + hotkey
  workspace.py         # per-conjecture run directories and prompt resolution
conjectures/
  algebra_example/
    conjecture.md          # required: the statement
    dag.json               # shared by every model
    dag.checkpoint.json    # optional: where a cancelled run stopped (--fresh)
    references/            # optional: source files (.tex/.md/plain text) to parse
    references.md          # parsed theorem-level results (strict JSON array)
    prover.md              # optional per-conjecture prompt override
flake.nix                  # Nix dev shell
```

### Conjectures and the shared DAG

`--conjecture` names a directory under `conjectures/` that must contain a
`conjecture.md`. The DAG is written beside it as `dag.json` and is *shared by
every model*: point a second model at a conjecture already under
way and it continues from the lemmas the first one proved. A lemma proved by
a 27B model and passed by the verifiers is no less proved when a 122B model
arrives. The cost is that this is collaboration, not a controlled comparison —
give a model its own file with `--dag` if you want them isolated again.

Prompt files are looked up in the conjecture directory first, then in
`agents/`, so a conjecture that needs a special prompt (e.g. a prover primed
for Brauer groups) can carry one without forking the defaults.

### Cancelling a run (Ctrl-C) and resuming

Ctrl-C stops the run and leaves a checkpoint — `<dag>.checkpoint.json`
beside `dag.json`, so a `--dag` override gets its own. It records what a
cancelled run would otherwise lose: where the iteration budget stood, the
planner's reject list (the `failed_attempts` the next planner call reads),
and, if the cancellation landed mid-proof, the lemma in flight, which prover
round it was on, and the feedback the earlier rounds of that attempt already
produced. The file is written at the safe boundaries — the top of each
iteration, before each prover round, and the end of each iteration — so it
always holds a coherent position, and it is deleted when the conjecture
resolves. An existing checkpoint is announced in the run banner.

Re-run the same command to pick up where the run stopped: the budget
continues rather than resets, and an in-flight lemma goes straight back to
the prover at the checkpointed round without asking the planner again. The
interrupted round itself is re-run — the checkpoint cannot reach into an
in-flight LLM call — but everything completed before it is kept. `--fresh`
deletes the checkpoint and restarts the budget from 1 without touching
`dag.json`; the DAG itself still needs deleting for a genuinely clean run.

A bad checkpoint degrades instead of crashing: an unreadable or incoherent
one is discarded and the run starts fresh; a checkpoint whose in-flight
lemma is already in the DAG drops only the in-flight part and resumes as a
normal iteration; one whose iteration stands past `--max-iterations` is
reported and kept — raise the budget and re-run to continue from it.

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
  is small: below `PLANNER_REFERENCE_LIMIT` (100) entries it gets every
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
