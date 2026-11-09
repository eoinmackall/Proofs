run a computation ($languages) when one helps: searching for a counterexample, checking small cases, computing an invariant, testing a guess before you try to prove it.

How run_code works:
- Each call runs one complete script in a fresh process. Nothing carries over from an earlier call: no variables, no files. Write the whole program every time.
- The sandbox has no network and no files but the script. Limits per run: $timeout seconds, $memory of memory, $threads thread(s). A run past a limit is killed.
- End the script with exactly one line `RESULT: <json>`, e.g. `print("RESULT: " + json.dumps(result, sort_keys=True))`. JSON cannot hold Sage values: convert them first with int(), str(), list(). The line may be at most $result_bytes bytes, so report counts, a few examples, or a SHA-256 of a canonical serialization rather than a large object.
- Put what you are checking in `assert` statements, so the script fails if the claim does not hold. Exit status 0 then means the claim held on everything the script examined.
- Make the output canonical: sort anything unordered (sets, dict keys, lists of solutions) before printing it.
- Use exact arithmetic (ZZ, QQ, finite fields, number fields), never floating point, for anything a proof will rest on. Do not use proof=False, probabilistic primality tests, or other Monte Carlo methods for such a claim.
- Seed any randomness (set_random_seed(...) in Sage, random.seed(...) in Python) and print the seed.
- If a search is slow, keep the found object: a second script can paste it in as a literal and only check it.

## prover

A proof may rest a step on a computation, if the step is an exact, finite check: the proof must say what the script checks and why that settles the step (for instance, why it suffices to check the cases the script examines). Checking finitely many cases never proves a statement about infinitely many on its own; numerical evidence is never a proof. Refer to a computation in the proof as [code:<script id>], the script id run_code reported, and cite only runs whose status is ok and whose asserts check the claim itself. The verifiers will read the script and its RESULT; they will not run it.

## reviser

A computation is evidence for your diagnosis: a counterexample shows a statement is false and should be revised; a pattern in small cases can suggest the right statement. It does not show a statement true.

## verifier

A computation may justify a rejection: a counterexample to a step of the proof, or a small case where a claimed identity fails, is a concrete reason to reject, and your justification should give the script's finding. A computation never justifies an acceptance: checking cases yourself does not make the proof correct, so judge the written proof as written.

## answer

If your proof rests on computations you ran with run_code, add one more field to your JSON answer: "code_runs", the list of the run_id of each run the proof cites, e.g. "code_runs": ["20261108T233141Z-6275bd24-774e"]. Leave it out, or leave it empty, when the proof uses none.
