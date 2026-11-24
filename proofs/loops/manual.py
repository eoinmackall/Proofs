"""Manual mode: the menu the planning step puts to the operator, and the
operator's own proof, checked by the verifiers like any other.

The menu (menu) runs at exactly one point in the loop — the planning step —
and only when the loop has already stopped and is waiting for a person:
"a couple of keystrokes" means `2` then Enter, and that is the entire
interaction; nothing downstream asks again. Its terminal input (ask,
read_block) and the hotkey that brings it up mid-run are ui/input.py's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
)

from ..core import resolution
from ..core.dag import _verified_hash
from ..ui.input import HotKey, ask, read_block
from ..validity import verdicts
from ..validity.citations import check_citations
from ..validity.steps import VERIFIER_AGENTS, role_of
from .prompts import _wrap, render, verifier_context
from .reasoning import VERIFIER_ROLE, _recorded_model, pinned
from .verifiers import _run_verifier, with_verifier_failover


def _operator_citation_check(
    dag: Any,
    ref_ids: Set[str],
    suspended: Set[str],
) -> Callable[[List[Any], List[Any]], Optional[str]]:
    """The menu's check_citations (see menu): the operator's
    citations, run through the prover's check, and the reason they cannot
    be used — an unknown or suspended lemma, a reference the collection
    does not hold — or None when every one resolves."""

    def check(raw_lemmas: List[Any], raw_refs: List[Any]) -> Optional[str]:
        _l, _r, unmatched = check_citations(
            raw_lemmas, raw_refs, dag, ref_ids, suspended
        )
        if not unmatched:
            return None
        why = {
            "lemma": "no such lemma in the DAG",
            "suspended": "suspended — refuted, uncertified, or standing on one that is",
            "reference": "no such lemma or reference",
            "other": "not a citation",
        }
        return "cannot cite " + "; ".join(
            f"{label} ({why.get(kind, kind)})" for label, kind in unmatched
        )

    return check


def _verify_operator_proof(
    ctx,
    choice: Choice,
    dag: Any,
    references: List[Dict[str, Any]],
    ref_ids: Set[str],
    suspended: Set[str],
    conjecture: str,
    verifier_sys: Dict[str, str],
    verbose: bool,
) -> Optional[Tuple[str, Dict[str, Any], Optional[str]]]:
    """The operator's own proof of a lemma (an "own_proof" Choice) through
    the three verifier checks, exactly as the proof loop puts the prover's:
    the conjecture, the statements of what the proof cites and nothing
    else, the target, the proof. No lemma enters the DAG on the operator's
    say-so; the operator stands in for the prover, not for the verifiers.

    Returns (lemma_id, node, verified_hash) for the caller to commit and
    certify when all three accept — the commit goes into the run's own
    stream, as every commit does — or None, the reason logged, when
    a citation does not resolve, a verifier rejects the proof (its
    objection is shown), or a verifier gives no verdict. The caller then
    puts the menu back."""
    lemma = resolution.pin(choice.lemma or {}, conjecture)
    lemma_id = str(lemma.get("id") or "").strip()
    statement = str(lemma.get("statement") or "").strip()
    proof = str(choice.proof or "").strip()
    if not lemma_id or not statement or not proof:
        ctx.ui.log("  That lemma has no usable id, statement or proof.", verbose)
        return None
    cited_lemmas, cited_refs, unmatched = check_citations(
        choice.cited_lemmas, choice.cited_references, dag, ref_ids, suspended
    )
    if unmatched:
        ctx.ui.log(
            "  Your proof cites results that cannot be used: "
            + ", ".join(label for label, _ in unmatched)
            + ". Nothing was verified.",
            verbose,
        )
        return None
    target = {"id": lemma_id, "statement": statement}
    ctx.ui.log(
        f"Verifying your proof of {lemma_id} ({len(proof)} chars"
        + (
            ", cites: " + ", ".join(
                str(d) for d in [*cited_lemmas, *cited_refs]
            )
            if cited_lemmas or cited_refs else ", no citations"
        )
        + ").",
        verbose,
    )
    verifier_user = (
        f"Conjecture:\n{conjecture}\n\n"
        f"Cited results (statements of exactly the lemmas and "
        f"references the proof declares it used; nothing else is "
        f"available to it):\n"
        f"{json.dumps(verifier_context(dag, cited_lemmas, cited_refs, references), indent=2)}\n\n"
        f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
        f"Proposed proof:\n{proof}"
    )
    ctx.status(lemma=lemma_id, attempt="")

    def run_check() -> bool:
        """The three checks; whether all accepted. Run again from the
        start on the fallback model when a verifier's call fails over
        (--with-fallback), so the certificate names one model."""
        for step, agent in enumerate(VERIFIER_AGENTS, 1):
            role = role_of(agent)
            decision, justification = _run_verifier(
                ctx,
                role, verifier_sys[agent], verifier_user, verbose,
            )
            ctx.ui.log(
                f"Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
                f"{decision.upper() or '???'} — {justification}",
                verbose,
            )
            if decision == verdicts.REJECT:
                ctx.ui.log(
                    f"Your proof of {lemma_id} was rejected by {role}; nothing "
                    f"was added. Back to the menu.",
                    verbose,
                )
                return False
            if decision != verdicts.ACCEPT:
                ctx.ui.log(
                    f"No verdict on your proof of {lemma_id} from {role} (the "
                    f"server failed or the reply could not be read); nothing "
                    f"was added. Back to the menu.",
                    verbose,
                )
                return False
        return True

    with pinned(ctx, {}):
        if not with_verifier_failover(ctx, run_check, verbose):
            return None
        # The model the check ran on, for the certificate the caller
        # issues: the fallback's, if the check was restarted there.
        verifier_model = _recorded_model(ctx, VERIFIER_ROLE)
    ctx.ui.log(
        f"Your proof of {lemma_id} passed all {len(VERIFIER_AGENTS)} "
        f"verifier checks. Adding to DAG.",
        verbose,
    )
    node = {
        "statement": statement,
        "proof": proof,
        "cited_lemmas": list(cited_lemmas),
        "cited_references": list(cited_refs),
        "_verifier_model": verifier_model,
    }
    return lemma_id, node, _verified_hash(dag, node)


def choose(ctx, *args: Any, **kwargs: Any) -> Choice:
    """The manual menu (menu), with the status board told the
    loop is waiting on the operator, and a live renderer paused while they
    read and type."""
    ctx.status(role="waiting for you", lemma="", attempt="")
    with ctx.ui.paused():
        return menu(*args, **kwargs)


@dataclass
class Choice:
    """What the operator decided at a planning step.

    action is one of:
        "own_proof" - the operator proves `lemma` themselves: `proof` is
                      their proof and `cited_lemmas` / `cited_references`
                      its citations, in the prover's raw shapes, already
                      checked by the menu's check_citations when it was
                      given one. The caller puts the proof through the
                      three verifiers, exactly as it would the prover's;
                      nothing enters the DAG on the operator's say-so
        "prove"     - send `lemma` down the usual prover/verifier pipeline
        "replan"    - discard all candidates, ask the planner again
        "auto"      - hand control back to the automation, starting now
        "quit"      - stop the run and keep the DAG as it stands

    "own_proof" and "prove" differ only in who writes the proof. Either
    way it is the verifiers who decide whether it goes in.
    """
    action: str
    lemma: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    proof: Optional[str] = None
    cited_lemmas: List[Any] = field(default_factory=list)
    cited_references: List[Any] = field(default_factory=list)


_MENU = (
    "  [1-9]  prove that one yourself — your proof goes to the verifiers\n"
    "  [1p]   send that one to the proving pipeline instead\n"
    "  [w]    write your own lemma and proof   [wp] write one and have it proved\n"
    "  [r]    none of these, re-plan           [a]  resume automation\n"
    "  [q]    quit and keep the DAG\n"
)


# A trailing 'p' is the whole grammar: it means "hand this to the models"
# wherever it appears. "3p" is the gesture as described — pick the candidate,
# then hit a key. "p3" reads more naturally to some people and costs nothing
# to accept, so both work, and likewise "wp" / "pw".
_PICK_RE = re.compile(r"(p?)(\d+)(p?)$")


# check_citations: given the raw cited_lemmas and cited_references, the
# reason they cannot be used (an unknown or suspended lemma, a reference the
# collection does not hold), or None when every citation resolves.
CitationCheck = Callable[[List[Any], List[Any]], Optional[str]]


def menu(
    shortlist: Sequence[Tuple[Dict[str, Any], List[str]]],
    new_id: Callable[[], str],
    hotkey: HotKey,
    plan_summary: str = "",
    check_citations: Optional[CitationCheck] = None,
) -> Choice:
    """Put the candidate list — each candidate with its notes — to the
    operator and return their decision.

    new_id draws the id of a lemma the operator writes (core.dag.new_lemma_id).
    check_citations, when given, is run on the citations of a proof the
    operator writes, before the menu returns it: a citation that does not
    resolve is reported and the citations asked for again, so a pasted
    proof is not lost to a typo in an id."""
    print("\n" + "─" * 72)
    print(f"Planner proposes {len(shortlist)} next steps:\n")
    print(render(shortlist, plan_summary))
    print("\n" + _MENU)

    with hotkey.paused():
        while True:
            answer = ask("> ")
            if answer is None:  # end of input at the menu: quit
                return Choice("quit")
            reply = answer.lower()

            if reply in ("q", "quit"):
                return Choice("quit")
            if reply in ("a", "auto"):
                return Choice("auto")
            if reply in ("r", "replan", ""):
                # End of input here is no reason given, not a reason "q".
                why = ask("reason for the planner (optional): ")
                note = why or "Rejected by the operator without a stated reason."
                return Choice("replan", notes=[note])
            if reply in ("w", "write", "wp", "pw", "writep"):
                lemma = _write_own(new_id)
                if lemma is None:
                    continue
                if reply in ("wp", "pw", "writep"):
                    return Choice("prove", lemma=lemma)
                own = _own_proof(lemma, check_citations)
                if own is None:
                    print(_MENU)
                    continue
                return own
            if reply in ("?", "h", "help"):
                print(_MENU)
                continue

            match = _PICK_RE.match(reply)
            if match and 1 <= int(match.group(2)) <= len(shortlist):
                cand, _ = shortlist[int(match.group(2)) - 1]
                if match.group(1) or match.group(3):
                    return Choice("prove", lemma=cand)
                own = _own_proof(cand, check_citations)
                if own is None:
                    print(_MENU)
                    continue
                return own

            print("  ? unrecognised. " + _MENU)


def _write_own(new_id: Callable[[], str]) -> Optional[Dict[str, Any]]:
    """Let the operator state the next lemma themselves.

    Worth having: the most valuable thing a mathematician can contribute to a
    run like this is usually not picking between the model's five ideas but
    supplying the sixth one it didn't think of. Whether the operator then
    proves it or the models do is the caller's business — `w` asks for the
    operator's proof next, `wp` sends it to the models, exactly as a digit
    and a digit with `p` do for a candidate.

    Only the statement is asked for here, as the planner supplies it; the
    lemma gets its id (new_id) the way a planner's candidate does. A
    statement of exactly "conjecture" or "conjecture_negation" attempts the
    conjecture, and the caller pins its statement (resolution.pin).
    """
    # End of input, an empty line or a lone "q" cancels the written lemma
    # and returns to the menu.
    statement = ask(
        "  statement (or \"conjecture\" / \"conjecture_negation\" to settle it): ")
    if not statement or statement == "q":
        print("  no statement; nothing added.")
        return None
    lemma = {"id": new_id(), "statement": statement}
    if statement in resolution.SETTLES:
        lemma["settles"] = statement
    print(f"  {lemma['id']}")
    return lemma


def _own_proof(
    lemma: Dict[str, Any], check_citations: Optional[CitationCheck]
) -> Optional[Choice]:
    """Ask the operator for a proof of `lemma` and its citations, and
    return them as an "own_proof" Choice — or None when they cancel, which
    puts the menu back.

    The proof is read with read_block (paste it, or type it; Enter
    submits). A pasted JSON object with a "proof" field — the prover's own
    output shape — is taken whole, its cited_lemmas and cited_references
    with it. Otherwise the citations are asked for on one line (see
    parse_citations). Citations that do not parse, or that check_citations
    finds do not resolve, are reported and asked for again; the proof is
    kept.
    """
    print(f"\n  Proof of {lemma.get('id', '?')}:")
    print(_wrap(str(lemma.get("statement", "")), indent="    "))
    proof = read_block(
        "  Paste or type your proof. Enter submits; Alt+Enter or Ctrl+J "
        "(or Shift+Enter, where your terminal sends it) starts a new line; "
        "Ctrl+D on an empty proof cancels.\n"
    )
    if proof is None or not proof.strip():
        print("  no proof; back to the menu.")
        return None

    cited: Optional[Tuple[List[Any], List[Any]]] = None
    as_json = _proof_object(proof)
    if as_json is not None:
        proof_text, raw = as_json
        try:
            cited = _citations_from_json(raw)
        except ValueError as e:
            print(f"  the pasted object's citations are not usable: {e}")
            return None
        proof = proof_text
        problem = check_citations(*cited) if check_citations else None
        if problem:
            print(f"  {problem}")
            cited = None  # ask for them on the line instead
    while cited is None:
        line = ask(
            "  cites (lemma and reference ids, comma-separated; or the "
            "prover's JSON; empty for none): "
        )
        if line is None or line == "q":
            print("  cancelled; back to the menu.")
            return None
        try:
            attempt = parse_citations(line)
        except ValueError as e:
            print(f"  {e}")
            continue
        problem = check_citations(*attempt) if check_citations else None
        if problem:
            print(f"  {problem}")
            continue
        cited = attempt
    return Choice(
        "own_proof", lemma=lemma, proof=proof.strip(),
        cited_lemmas=cited[0], cited_references=cited[1],
    )


# An id as the files spell one: a user_id, a lemma_id or a reference id.
_ID_RE = re.compile(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*$")


def parse_citations(text: str) -> Tuple[List[Any], List[Any]]:
    """The operator's citations line as (cited_lemmas, cited_references),
    in the prover's raw shapes.

    Either the prover's JSON — an object with "cited_lemmas" and
    "cited_references" lists of ids, or a bare list of ids — or the short
    form: ids separated by commas or spaces, a lemma's id (it has slashes,
    user/run/n) and a reference's alike. An empty line is no citations.
    Anything else raises ValueError saying what is wrong; nothing is
    guessed. Whether the ids name results that exist, and which kind, is
    check_citations' to say (see menu()).
    """
    text = text.strip()
    if not text:
        return [], []
    if text[0] in "{[":
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise ValueError(f"not valid JSON: {e}") from None
        return _citations_from_json(raw)
    ids: List[Any] = []
    for token in re.split(r"[,\s]+", text):
        if not token:
            continue
        if not _ID_RE.match(token):
            raise ValueError(f"{token!r} is not an id")
        ids.append(token)
    return [i for i in ids if "/" in i], [i for i in ids if "/" not in i]


def _citations_from_json(raw: Any) -> Tuple[List[Any], List[Any]]:
    """Citations in the prover's JSON shapes, checked for shape only: an
    object with list-valued "cited_lemmas" / "cited_references" (either may
    be missing), or a list of ids."""
    if isinstance(raw, dict):
        unknown = set(raw) - {"cited_lemmas", "cited_references", "proof"}
        if unknown:
            raise ValueError(
                f"unexpected key(s) {', '.join(sorted(unknown))}; the "
                f"citations are cited_lemmas and cited_references"
            )
        lemmas = raw.get("cited_lemmas", [])
        refs = raw.get("cited_references", [])
        if not isinstance(lemmas, list) or not isinstance(refs, list):
            raise ValueError("cited_lemmas and cited_references must be lists")
        entries = [*lemmas, *refs]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("expected a JSON object or list of citations")
    for entry in entries:
        if not (isinstance(entry, str) and _ID_RE.match(entry)):
            raise ValueError(f"{json.dumps(entry)} is not an id")
    return [e for e in entries if "/" in e], [e for e in entries if "/" not in e]


def _proof_object(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(proof, the object) when the pasted text is the prover's JSON — an
    object with a string "proof" — else None, and the text is the proof
    itself."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("proof"), str):
        return obj["proof"], obj
    return None
