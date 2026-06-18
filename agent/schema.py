"""Schema-rendering helper.

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.

When a `question` is provided, the schema is pruned to a question-relevant
subset (top-k by question/identifier keyword overlap, expanded by one-hop
foreign-key neighbors). This is the Phase 6 / Iter 10 fix that prevents
the worst-case DBs (`debit_card_specializing`, `european_football_2`) from
shipping ~5 K-token schemas in every prompt, which dominated KV slot
occupancy and made `max-num-seqs` impossible to size.
"""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"

# Top-k tables to keep after question-keyword scoring (pre FK expansion).
SCHEMA_PRUNE_TOP_K = 6


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _tokenize(text: str) -> set[str]:
    """Lowercase alphabetic tokens of length >2; drops digits and short words."""
    return {tok for tok in re.findall(r"[a-z]+", text.lower()) if len(tok) > 2}


def _render_subset(conn: sqlite3.Connection, db_id: str, tables: list[str]) -> str:
    """Render CREATE TABLE text for the given subset of tables."""
    parts: list[str] = [f"-- Database: {db_id}"]
    for t in tables:
        parts.append(f"\nCREATE TABLE {_q(t)} (")
        col_lines: list[str] = []
        for _cid, name, ctype, notnull, _dflt, pk in conn.execute(
            f"PRAGMA table_info({_q(t)})"
        ):
            line = f"  {_q(name)} {ctype}"
            if pk:
                line += " PRIMARY KEY"
            if notnull and not pk:
                line += " NOT NULL"
            col_lines.append(line)
        for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
            # (id, seq, ref_table, from, to, on_update, on_delete, match)
            ref = f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}"
            if fk[4] is not None:
                ref += f"({_q(fk[4])})"
            col_lines.append(ref)
        parts.append(",\n".join(col_lines))
        parts.append(");")
    return "\n".join(parts)


@lru_cache(maxsize=512)
def render_schema(db_id: str, question: str = "") -> str:
    """Render CREATE TABLE text, optionally pruned to tables relevant to `question`.

    Pruning rule:
      1. Tokenize the question into alphabetic word tokens (length >2, lowercase).
      2. For each table, score = number of question tokens that appear as tokens
         in the table name or any of its column names.
      3. Keep the top-k tables by (score desc, name asc) for determinism.
      4. Expand by FK neighbors in both directions (one hop) so join paths are
         preserved.
      5. If `question` is empty OR every kept table scored 0 (no useful signal),
         fall back to rendering the full schema.

    The (db_id, question) pair is cached so repeated identical calls (e.g. the
    same question's generate/verify/revise calls in one agent run) are free.
    """
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(
            f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?"
        )

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        all_tables: list[str] = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]

        # Fast paths: nothing to prune, or no question to prune against.
        if not question.strip() or len(all_tables) <= SCHEMA_PRUNE_TOP_K:
            return _render_subset(conn, db_id, all_tables)

        # Score tables by keyword overlap.
        q_tokens = _tokenize(question)
        scores: dict[str, int] = {}
        for t in all_tables:
            name_tokens = _tokenize(t)
            for row in conn.execute(f"PRAGMA table_info({_q(t)})"):
                name_tokens |= _tokenize(row[1])
            scores[t] = len(q_tokens & name_tokens)

        # Sort: score desc, name asc (deterministic tie-break).
        ranked = sorted(all_tables, key=lambda t: (-scores[t], t))
        top = ranked[:SCHEMA_PRUNE_TOP_K]

        # If nothing scored at all, fall back to the full schema — don't guess.
        if all(scores[t] == 0 for t in top):
            return _render_subset(conn, db_id, all_tables)

        kept: set[str] = set(top)
        all_tables_set = set(all_tables)

        # Forward FK expansion: tables that kept tables reference.
        for t in list(kept):
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                ref_table = fk[2]
                if ref_table in all_tables_set:
                    kept.add(ref_table)

        # Reverse FK expansion: tables that reference any kept table.
        for t in all_tables:
            if t in kept:
                continue
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                if fk[2] in kept:
                    kept.add(t)
                    break

        return _render_subset(conn, db_id, sorted(kept))


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
