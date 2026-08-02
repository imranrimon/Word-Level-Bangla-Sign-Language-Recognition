"""Extract the 401 word names from BdSLW401's `bdsl words-complete.pdf`.

The PDF is a 4-column table:
    Word Number | Word in Bangla | Word in Romanised Bangla | Word Meaning in English

We parse it via pypdf, join wrapped lines, and emit a JSON with one entry
per word id (W001..W401):

    {
      "W001": {
        "bangla":    "বাবা /আব্বা",
        "romanized": "Baba/Abba",
        "english":   "Father"
      },
      ...
    }

The Romanized column is the canonical key for vocabulary alignment with
BdSLW60 (whose class folders are romanized Bangla like 'aam', 'amma').

Optional second output: regenerate `data/bdslw401_si/classes.json` with
romanized names as class IDs (rather than the W001..W401 numeric IDs the
bundler currently uses), so `build_bangla_vocab_alignment.py` can do
lexical matching across datasets directly.

Usage:

    python preprocessing/extract_bdslw401_word_names.py \\
        --pdf "data/bdslw401_raw/bdsl words-complete.pdf" \\
        --output data/bdslw401_words.json \\
        --rewrite-classes data/bdslw401_si/classes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


# A row starts with W<digits> followed by whitespace.
_ROW_START = re.compile(r"^W(\d{1,4})\s+", re.IGNORECASE)


def _is_bangla_token(s):
    return any(0x0980 <= ord(c) <= 0x09FF for c in s)


def _split_columns(joined_line):
    """Split a joined 'W<id> bangla romanized english' line into 4 columns.

    Strategy: the Bangla column has Bangla glyphs; the Romanized column has
    only ASCII letters/digits/slash; the English column comes last and may
    contain spaces.

    Token stream: ['W001', '<bangla...>', '<romanized...>', '<english...>']
    """
    parts = joined_line.split()
    if not parts or not _ROW_START.match(parts[0] + " "):
        return None
    wid = parts[0]
    rest = parts[1:]

    # Bangla phase: take consecutive tokens that contain any Bangla codepoint.
    i = 0
    bangla_tokens = []
    while i < len(rest) and _is_bangla_token(rest[i]):
        bangla_tokens.append(rest[i])
        i += 1
    # If no Bangla token, the OCR mangled this row — skip.
    if not bangla_tokens:
        return None

    # Romanized phase: the BdSLW401 PDF's Romanized column is always a
    # SINGLE token (possibly slash-joined, e.g., "Baba/Abba" or "Dada/Nana").
    # English meaning follows and may contain spaces / parentheses.
    romanized_tokens = []
    if i < len(rest) and not _is_bangla_token(rest[i]):
        romanized_tokens = [rest[i]]
        i += 1

    english_tokens = rest[i:]
    return {
        "id": wid.upper(),
        "bangla": " ".join(bangla_tokens),
        "romanized": " ".join(romanized_tokens).strip("/").strip(),
        "english": " ".join(english_tokens).strip(),
    }


def parse_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_lines = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        raw_lines.extend(l.rstrip() for l in txt.splitlines())

    # Group: a new entry begins on a line that matches _ROW_START. Subsequent
    # lines belong to the previous entry until the next match.
    groups = []
    current = []
    for line in raw_lines:
        if _ROW_START.match(line):
            if current:
                groups.append(" ".join(current))
            current = [line]
        else:
            if current:
                current.append(line.strip())
    if current:
        groups.append(" ".join(current))

    entries = {}
    for g in groups:
        parsed = _split_columns(g)
        if parsed is None:
            continue
        entries[parsed["id"]] = {
            "bangla": parsed["bangla"],
            "romanized": parsed["romanized"],
            "english": parsed["english"],
        }
    return entries


def _normalise_id(wid, total_classes=401):
    """Pad to 3 digits with W prefix. 'W1' -> 'W001'."""
    n = int(re.sub(r"\D", "", wid))
    width = max(3, len(str(total_classes)))
    return f"W{n:0{width}d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", default="data/bdslw401_raw/bdsl words-complete.pdf")
    ap.add_argument("--output", default="data/bdslw401_words.json")
    ap.add_argument(
        "--rewrite-classes", default=None,
        help="optionally regenerate this classes.json to use romanized "
             "names as the class IDs (preserves the original 0-indexed order)",
    )
    args = ap.parse_args()

    entries = parse_pdf(args.pdf)
    # Normalise ids to W001..W401.
    normalised = {_normalise_id(k): v for k, v in entries.items()}
    print(f"parsed {len(normalised)} entries from {args.pdf}")

    out = {
        "source_pdf": args.pdf,
        "n_entries": len(normalised),
        "entries": dict(sorted(normalised.items())),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")

    # Show first 10 entries for spot-check.
    print("\nfirst 10 entries:")
    for wid, e in list(out["entries"].items())[:10]:
        print(f"  {wid}: bangla={e['bangla']!r}, romanized={e['romanized']!r}, english={e['english']!r}")

    if args.rewrite_classes:
        cls_path = Path(args.rewrite_classes)
        if not cls_path.is_file():
            raise SystemExit(f"--rewrite-classes target not found: {cls_path}")
        with open(cls_path, encoding="utf-8") as f:
            data = json.load(f)
        wids = data["classes"]                            # ['W001', 'W002', ...]
        romanized_names = []
        missing = []
        for wid in wids:
            entry = normalised.get(wid)
            if entry and entry["romanized"]:
                romanized_names.append(entry["romanized"])
            else:
                missing.append(wid)
                romanized_names.append(wid)               # fall back to numeric
        # Write a SIBLING file rather than overwriting (safer).
        out_path = cls_path.parent / "classes_romanized.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": romanized_names,
                "class_to_idx": {c: i for i, c in enumerate(romanized_names)},
                "original_class_ids": wids,
                "missing_word_names": missing,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nrewrote with romanized names -> {out_path}")
        if missing:
            print(f"  [warn] {len(missing)} class IDs had no PDF entry: "
                  f"first 5 = {missing[:5]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
