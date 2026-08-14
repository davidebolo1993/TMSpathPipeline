#!/usr/bin/env python3
"""
fix_fstrings.py -- make Python 3.12+ f-strings parse on older Pythons.

Python 3.12 (PEP 701) allows a quote character to be reused inside its own
f-string.  Older interpreters do not:

    f'{json_data['seedChans']}'      # 3.12+ only  -> SyntaxError on <= 3.11
    f"{json_data["seedChans"]}"      # 3.12+ only  -> SyntaxError on <= 3.11

This rewrites such f-strings to the portable form, preferring a double-quoted
outer literal with single-quoted expressions:

    f"{json_data['seedChans']}"      # runs everywhere

Usage
-----
    python3 fix_fstrings.py FILE [FILE ...]        # rewrite in place (+ .bak)
    python3 fix_fstrings.py --check FILE           # report only, exit 1 if work
    python3 fix_fstrings.py --diff  FILE           # print a unified diff
    python3 fix_fstrings.py --no-backup FILE
    python3 fix_fstrings.py --verify-with /usr/bin/python3 FILE
    python3 fix_fstrings.py scripts/              # recurse: *.py and *.ipynb

Accepts .py and .ipynb (code cells are fixed in place inside the notebook).

IMPORTANT: run this tool ON PYTHON 3.12 OR NEWER.  It needs the 3.12 tokenizer
to read the very syntax it is converting -- an older interpreter cannot even
parse the input file.  The *output* is what runs on the old machine.

As a safety net every rewrite is gated on `ast.dump()` being byte-identical
before and after, so quoting changes but no string VALUE ever does.  Anything
that cannot be proven safe is left untouched and reported, never guessed at.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import json
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

PREFIX_CHARS = "fFrRbBuU"

if sys.version_info < (3, 12):
    sys.exit(
        f"fix_fstrings.py needs Python 3.12+ to run (found {sys.version.split()[0]}).\n"
        "It uses the 3.12 tokenizer to read PEP 701 f-strings; older interpreters\n"
        "cannot parse the input at all.  Run it on the new Mac, copy the output."
    )


# --------------------------------------------------------------------------- #
# report types
# --------------------------------------------------------------------------- #

@dataclass
class Note:
    line: int
    reason: str
    snippet: str


@dataclass
class Report:
    converted: int = 0          # outer '...' -> "..."
    inner_flipped: int = 0      # inner "..." -> '...'
    skipped: list[Note] = field(default_factory=list)   # left alone, still valid
    manual: list[Note] = field(default_factory=list)    # still broken, needs a human

    @property
    def changed(self) -> bool:
        return bool(self.converted or self.inner_flipped)

    def summary(self, path: str) -> str:
        bits = [
            f"{path}: converted {self.converted} f-string(s), "
            f"flipped {self.inner_flipped} inner literal(s)"
        ]
        for n in self.skipped:
            bits.append(f"  skipped  line {n.line}: {n.reason}\n             {n.snippet}")
        for n in self.manual:
            bits.append(f"  MANUAL   line {n.line}: {n.reason}\n             {n.snippet}")
        return "\n".join(bits)


# --------------------------------------------------------------------------- #
# tokenizing helpers
# --------------------------------------------------------------------------- #

@dataclass
class FStr:
    """One f-string literal and the tokens that live inside it."""
    start: int
    prefix: str
    quote: str
    depth: int
    end: int = -1
    middles: list[str] = field(default_factory=list)      # literal-text segments
    strings: list[tuple[int, int, str]] = field(default_factory=list)  # inner STRINGs
    nested: int = 0                                       # nested f-string count
    line: int = 0


def _line_offsets(src: str) -> list[int]:
    offs = [0]
    for ln in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(ln))
    return offs


def _scan(src: str) -> list[FStr]:
    """Return every f-string literal in `src`, outermost ones at depth 0."""
    offs = _line_offsets(src)

    def pos(rc: tuple[int, int]) -> int:
        return offs[rc[0] - 1] + rc[1]

    out: list[FStr] = []
    stack: list[FStr] = []

    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.FSTRING_START:
            raw = t.string
            prefix = raw.rstrip("\"'")
            rec = FStr(
                start=pos(t.start),
                prefix=prefix,
                quote=raw[len(prefix):],
                depth=len(stack),
                line=t.start[0],
            )
            if stack:
                stack[-1].nested += 1
            stack.append(rec)
            out.append(rec)

        elif t.type == tokenize.FSTRING_END:
            stack.pop().end = pos(t.end)

        elif t.type == tokenize.FSTRING_MIDDLE:
            if stack:
                stack[-1].middles.append(t.string)

        elif t.type == tokenize.STRING and stack:
            stack[-1].strings.append((pos(t.start), pos(t.end), t.string))

    return out


def _split_prefix(raw: str) -> tuple[str, str, str]:
    """'rb"x"' -> ('rb', '"', 'x')  (prefix, quote, body)."""
    prefix = raw[: len(raw) - len(raw.lstrip(PREFIX_CHARS))]
    rest = raw[len(prefix):]
    quote = rest[:3] if rest[:3] in ('"""', "'''") else rest[:1]
    return prefix, quote, rest[len(quote): -len(quote)]


# --------------------------------------------------------------------------- #
# the fix
# --------------------------------------------------------------------------- #

def fix_source(src: str) -> tuple[str, Report]:
    """Return (fixed_source, report).  Never raises on odd input; reports instead."""
    rep = Report()
    edits: list[tuple[int, int, str]] = []

    for rec in _scan(src):
        if rec.depth != 0:
            continue  # handled via its parent, or left alone

        raw = src[rec.start: rec.end]
        snippet = raw if len(raw) <= 110 else raw[:107] + "..."
        is_raw = "r" in rec.prefix.lower()
        target = '"' * len(rec.quote)          # '  -> "   and  ''' -> """

        # Which inner string literals would collide with a double-quoted outer?
        collide = [s for s in rec.strings if _split_prefix(s[2])[1].startswith('"')]

        # ---- reasons we must not touch this one --------------------------- #
        if rec.nested:
            rep.skipped.append(Note(rec.line, "contains a nested f-string", snippet))
            continue

        if any('"' in m for m in rec.middles):
            # e.g. f'he said "hi"'.  Already legal on old Python; converting
            # would mean backslash-escaping, which is churn for no gain.
            if rec.quote == "'":
                rep.skipped.append(
                    Note(rec.line, 'literal text contains " (already portable)', snippet)
                )
                continue

        bad = None
        for _s, _e, text in collide:
            p, q, body = _split_prefix(text)
            if "'" in body:
                bad = f"inner literal {text!r} contains both quote kinds"
            elif "r" in p.lower() and "\\" in body:
                bad = f"inner raw literal {text!r} cannot be requoted"
        if bad:
            rep.manual.append(Note(rec.line, bad, snippet))
            continue

        # ---- apply --------------------------------------------------------- #
        outer_needs_change = rec.quote.startswith("'")

        if outer_needs_change:
            if rec.quote == "'''" and ('"""' in "".join(rec.middles)
                                       or "".join(rec.middles).endswith('"')):
                rep.skipped.append(Note(rec.line, "triple-quote body blocks swap", snippet))
                continue
            if is_raw and any('"' in m for m in rec.middles):
                rep.skipped.append(Note(rec.line, "raw f-string cannot be escaped", snippet))
                continue

        if not outer_needs_change and not collide:
            continue  # already f"..." with no inner conflict: nothing to do

        if outer_needs_change:
            edits.append((rec.start, rec.start + len(rec.prefix) + len(rec.quote),
                          rec.prefix + target))
            edits.append((rec.end - len(rec.quote), rec.end, target))
            rep.converted += 1

        for s, e, text in collide:
            p, q, body = _split_prefix(text)
            newq = "'" * len(q)
            edits.append((s, e, p + newq + body + newq))
            rep.inner_flipped += 1

    if not edits:
        return src, rep

    out, last = [], 0
    for s, e, replacement in sorted(edits):
        if s < last:                       # overlapping edits: bail out safely
            raise RuntimeError(f"overlapping edits at offset {s}")
        out.append(src[last:s])
        out.append(replacement)
        last = e
    out.append(src[last:])
    return "".join(out), rep


def assert_same_ast(before: str, after: str) -> None:
    """Hard gate: quoting may change, values may not."""
    if ast.dump(ast.parse(before)) != ast.dump(ast.parse(after)):
        raise RuntimeError("AST changed -- refusing to write")


def remaining_problems(src: str, whole: str | None = None) -> list[Note]:
    """f-strings that still will not parse before 3.12, for any reason."""
    whole = src if whole is None else whole
    bad: list[Note] = []
    for rec in _scan(src):
        raw = src[rec.start: rec.end]
        snippet = raw if len(raw) <= 110 else raw[:107] + "..."

        # (a) a quote character reused inside its own f-string
        q = rec.quote[0]
        for _s, _e, text in rec.strings:
            if _split_prefix(text)[1].startswith(q):
                bad.append(Note(rec.line, "still reuses its own quote", snippet))
                break

        # (b) a backslash anywhere in a replacement field.  Illegal before 3.12
        # however it is quoted, so requoting cannot rescue it -- needs a human.
        # Backslashes in the *literal* text are fine, so subtract those out.
        if raw.count("\\") > sum(m.count("\\") for m in rec.middles):
            bad.append(Note(rec.line, "backslash in an expression (needs a temp variable)",
                            snippet))
    return bad


# --------------------------------------------------------------------------- #
# file / notebook plumbing
# --------------------------------------------------------------------------- #

def fix_python(path: Path) -> tuple[str, str, Report]:
    src = path.read_text(encoding="utf-8")
    new, rep = fix_source(src)
    if rep.changed:
        assert_same_ast(src, new)
    return src, new, rep


def fix_notebook(path: Path) -> tuple[str, str, Report]:
    src = path.read_text(encoding="utf-8")
    nb = json.loads(src)
    total = Report()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell["source"])
        try:
            new_code, rep = fix_source(code)
        except (SyntaxError, tokenize.TokenError):
            continue  # cell has magics/incomplete code -- leave it alone
        if rep.changed:
            assert_same_ast(code, new_code)
            cell["source"] = new_code.splitlines(keepends=True)
        total.converted += rep.converted
        total.inner_flipped += rep.inner_flipped
        total.skipped += rep.skipped
        total.manual += rep.manual
    return src, json.dumps(nb, indent=1, ensure_ascii=False) + "\n", total


def verify_with(interpreter: str, source: str, name: str) -> str | None:
    """Compile `source` with another interpreter; return its error, or None."""
    proc = subprocess.run(
        [interpreter, "-c",
         "import sys; compile(sys.stdin.read(), sys.argv[1], 'exec')", name],
        input=source, capture_output=True, text=True,
    )
    return None if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1]


def iter_targets(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*")
                              if q.suffix in {".py", ".ipynb"} and q.is_file())
        else:
            yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files or directories (.py / .ipynb)")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if work remains")
    ap.add_argument("--diff", action="store_true", help="print a unified diff, do not write")
    ap.add_argument("--no-backup", action="store_true", help="do not leave a .bak file")
    ap.add_argument("--verify-with", metavar="PYTHON",
                    help="also compile the result with this interpreter, e.g. /usr/bin/python3")
    args = ap.parse_args(argv)

    exit_code = 0
    for path in iter_targets(args.paths):
        try:
            handler = fix_notebook if path.suffix == ".ipynb" else fix_python
            src, new, rep = handler(path)
        except (SyntaxError, tokenize.TokenError, RuntimeError) as exc:
            print(f"{path}: SKIPPED -- {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        print(rep.summary(str(path)))

        if path.suffix != ".ipynb":
            for note in remaining_problems(new):
                print(f"  STILL BROKEN line {note.line}: {note.reason}\n"
                      f"             {note.snippet}")
                exit_code = 1

        if args.verify_with and rep.changed and path.suffix == ".py":
            err = verify_with(args.verify_with, new, str(path))
            if err:
                print(f"  {args.verify_with} REJECTS the result: {err}", file=sys.stderr)
                exit_code = 1
            else:
                print(f"  verified: compiles with {args.verify_with}")

        if args.diff:
            sys.stdout.writelines(difflib.unified_diff(
                src.splitlines(keepends=True), new.splitlines(keepends=True),
                f"a/{path}", f"b/{path}"))
            continue

        if args.check:
            if rep.changed:
                exit_code = 1
            continue

        if rep.changed:
            if not args.no_backup:
                path.with_suffix(path.suffix + ".bak").write_text(src, encoding="utf-8")
            path.write_text(new, encoding="utf-8")
            print(f"  written{'' if args.no_backup else f' (backup: {path.name}.bak)'}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
