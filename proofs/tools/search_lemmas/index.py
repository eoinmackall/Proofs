"""The search index: every proved statement, by the words in it.

An in-memory SQLite FTS5 table (SQLite's own full-text index, in the
standard library's sqlite3), built in a background thread the first time
search_lemmas is offered, from the DAG as it stands then, and kept up to
date by the run's commits (core/dag.py, _append_version). It is never
written to disk: every run builds its own.

The index holds row numbers and the words of each statement, not the
statements themselves (a contentless table): the statements are in the
DAG, and a search reads them from the caller's snapshot. So the index
only proposes: what it returns is filtered against the snapshot the agent
was shown, which may be older than the index.

A query is never handed to FTS5's own query syntax, where a model's
backslash, hyphen or colon is a syntax error. It is split into units — a
"quoted phrase", a word, or a word ending in * (a prefix) — each unit
quoted, and the units joined with OR and ranked by BM25 (_expression), so
a lemma matching more of the rarer words ranks first. A unit is split
into words by the index's own tokenizer, so the query and the statements
are read the same way: case and accents folded, LaTeX's backslashes and
braces dropped (\\mathbb{F}_q reads as mathbb f q), English endings
stemmed (curves, curve).

A word found in more than PRUNE_FRACTION of all statements is dropped
from the query — the query's KEEP_RAREST rarest units always stay —
since matching it means visiting most of the index and it does next to
nothing for the ranking. The word counts are read when the build ends, and
each commit adds its statement's words (put), so a word only this run's
lemmas use is found too; a reproof's old words are not taken off — there
are too few to matter.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Callable, Dict, List, Mapping, Optional, Tuple

TOKENIZER = "porter unicode61"
PRUNE_FRACTION = 0.20
KEEP_RAREST = 3

# A unit of a query: a "quoted phrase", or a run of non-space characters.
_UNIT = re.compile(r'"([^"]*)"|(\S+)')


def _connect() -> sqlite3.Connection:
    """An in-memory database holding the index table (lemmas) and its word
    counts (vocab). Contentless, so the statements are not stored twice,
    where SQLite is new enough (3.43) to delete from such a table; a plain
    table otherwise. sqlite3.OperationalError when SQLite has no FTS5."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE lemmas USING fts5(statement, "
            f"tokenize='{TOKENIZER}', content='', contentless_delete=1)")
    except sqlite3.OperationalError as e:
        if "fts5" in str(e).lower() and "no such module" in str(e).lower():
            raise
        conn.execute(f"CREATE VIRTUAL TABLE lemmas USING fts5(statement, "
                     f"tokenize='{TOKENIZER}')")
    conn.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(lemmas, 'row')")
    return conn


def _scratch() -> sqlite3.Connection:
    """A table with the index's tokenizer, for splitting query units into
    the words the index holds: each word with its position (instance)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(f"CREATE VIRTUAL TABLE q USING fts5(s, tokenize='{TOKENIZER}')")
    conn.execute("CREATE VIRTUAL TABLE qv USING fts5vocab(q, 'instance')")
    return conn


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


class LemmaIndex:
    """The index of one run (Context.run.search_index), shared by its
    threads. start() returns at once; the build runs in a thread, and
    search() waits for it. error says why there is no index — no FTS5 in
    this SQLite, or a build that failed — and None while there is one, or
    may yet be."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._conn: Optional[sqlite3.Connection] = None
        self._scratch: Optional[sqlite3.Connection] = None
        self._ids: List[str] = []           # row number -> lemma id
        self._rows: Dict[str, int] = {}     # lemma id -> row number
        self._doc_freq: Dict[str, int] = {}
        # Commits made while the build runs, applied when it ends.
        self._pending: Dict[str, str] = {}
        self.error: Optional[str] = None
        self.reported = False

    @classmethod
    def start(cls, lemmas: Mapping[str, Mapping[str, object]]) -> "LemmaIndex":
        """An index of lemmas (a DAG's, keyed by id), building in the
        background. The caller holds the DAG lock, so no commit falls
        between the DAG lemmas came from and the first put()."""
        index = cls()
        threading.Thread(target=index._build, args=(lemmas,),
                         name="search-index", daemon=True).start()
        return index

    def failed(self) -> bool:
        """The build has ended without an index."""
        return self._ready.is_set() and self.error is not None

    def wait(self) -> Optional[str]:
        """Wait for the build; the error, when it ended without an index."""
        self._ready.wait()
        return self.error

    def _build(self, lemmas: Mapping[str, Mapping[str, object]]) -> None:
        try:
            conn = _connect()
            ids = list(lemmas)
            conn.executemany(
                "INSERT INTO lemmas(rowid, statement) VALUES (?, ?)",
                ((n, str(lemmas[lid].get("statement") or ""))
                 for n, lid in enumerate(ids)))
            doc_freq = dict(conn.execute("SELECT term, doc FROM vocab"))
            scratch = _scratch()
            with self._lock:
                self._conn, self._scratch = conn, scratch
                self._ids, self._doc_freq = ids, doc_freq
                self._rows = {lid: n for n, lid in enumerate(ids)}
                for lid, statement in self._pending.items():
                    self._put_locked(lid, statement)
                self._pending.clear()
        except sqlite3.OperationalError as e:
            self.error = (f"this Python's SQLite has no FTS5 ({e})"
                          if "fts5" in str(e).lower() else f"the build failed: {e}")
        except Exception as e:  # a failed build loses the tool, not the run
            self.error = f"the build failed: {e}"
        finally:
            self._ready.set()

    # -- commits ------------------------------------------------------------

    def put(self, lemma_id: str, statement: str) -> None:
        """A commit: a new lemma, or a new version of one, whose statement
        replaces the old one's words."""
        with self._lock:
            if self._conn is None:
                if self.error is None:
                    self._pending[lemma_id] = statement
                return
            self._put_locked(lemma_id, statement)

    def _put_locked(self, lemma_id: str, statement: str) -> None:
        row = self._rows.get(lemma_id)
        if row is None:
            row = len(self._ids)
            self._ids.append(lemma_id)
            self._rows[lemma_id] = row
        else:
            self._conn.execute("DELETE FROM lemmas WHERE rowid = ?", (row,))
        self._conn.execute(
            "INSERT INTO lemmas(rowid, statement) VALUES (?, ?)", (row, statement))
        self._scratch.execute("INSERT INTO q(rowid, s) VALUES (0, ?)", (statement,))
        for (term,) in self._scratch.execute("SELECT DISTINCT term FROM qv"):
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._scratch.execute("DELETE FROM q")

    # -- queries ------------------------------------------------------------

    def _units(self, query: str) -> List[Tuple[str, bool, List[str]]]:
        """The query's units as (text, is a prefix, the index words in
        it), those with no word left out (a lone $ or =)."""
        raw: List[Tuple[str, bool]] = []
        for phrase, chunk in _UNIT.findall(query):
            if phrase:
                raw.append((phrase, False))
            elif len(chunk) > 1 and chunk.endswith("*"):
                raw.append((chunk[:-1], True))
            else:
                raw.append((chunk, False))
        if not raw:
            return []
        scratch = self._scratch
        scratch.executemany("INSERT INTO q(rowid, s) VALUES (?, ?)",
                            ((n, text) for n, (text, _p) in enumerate(raw)))
        words: Dict[int, List[Tuple[int, str]]] = {}
        for term, doc, offset in scratch.execute("SELECT term, doc, offset FROM qv"):
            words.setdefault(doc, []).append((offset, term))
        scratch.execute("DELETE FROM q")
        return [(text, prefix, [t for _o, t in sorted(words[n])])
                for n, (text, prefix) in enumerate(raw) if n in words]

    def _expression(self, query: str, prune: bool) -> Optional[str]:
        """The FTS5 query for query (see the module docstring): its units,
        quoted, joined with OR. A unit holding a word no statement has
        cannot match and is left out; prune drops the common ones. None
        when nothing is left to search for."""
        kept: List[Tuple[str, bool]] = []
        counted: List[Tuple[int, str, bool]] = []
        for text, prefix, words in self._units(query):
            if prefix:              # its count is the sum of every completion
                kept.append((text, True))
                continue
            freq = min(self._doc_freq.get(w, 0) for w in words)
            if freq:
                counted.append((freq, text, False))
        counted.sort(key=lambda c: c[0])
        limit = PRUNE_FRACTION * len(self._ids) if prune else float("inf")
        for n, (freq, text, prefix) in enumerate(counted):
            if freq <= limit or n < KEEP_RAREST:
                kept.append((text, prefix))
        if not kept:
            return None
        return " OR ".join(_quote(text) + ("*" if prefix else "")
                           for text, prefix in kept)

    def search(self, query: str, want: Optional[int],
               accept: Callable[[str], bool]) -> List[str]:
        """The ids of the lemmas matching query, best first, that accept
        takes — at least want of them when there are that many (None for
        every one). The ranking is re-run with a larger limit when accept
        turns down too many of a batch."""
        with self._lock:
            expression = self._expression(query, prune=True)
            if expression is None:
                return []
            batch = None if want is None else 2 * want + 10
            while True:
                sql = "SELECT rowid FROM lemmas WHERE lemmas MATCH ? ORDER BY bm25(lemmas)"
                rows = (self._conn.execute(sql, (expression,)).fetchall()
                        if batch is None else
                        self._conn.execute(sql + " LIMIT ?", (expression, batch)).fetchall())
                out = [lid for lid in (self._ids[r] for (r,) in rows) if accept(lid)]
                if batch is None or len(out) >= want or len(rows) < batch:
                    return out
                batch *= 4

    def rank(self, query: str, statements: Mapping[str, str]) -> List[str]:
        """The ids of statements (id -> statement) that match query, best
        first, ranked among themselves: for a search narrowed to a few
        lemmas (the citers of one), where a temporary index of just those
        is quicker than the whole index filtered."""
        with self._lock:
            expression = self._expression(query, prune=False)
        if expression is None or not statements:
            return []
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(s, tokenize='{TOKENIZER}')")
            ids = list(statements)
            conn.executemany("INSERT INTO t(rowid, s) VALUES (?, ?)",
                             ((n, statements[lid]) for n, lid in enumerate(ids)))
            return [ids[r] for (r,) in conn.execute(
                "SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t)", (expression,))]
        finally:
            conn.close()
