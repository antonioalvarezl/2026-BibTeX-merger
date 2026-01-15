#!/usr/bin/env python3
"""Merge duplicate BibTeX entries using bibtexparser for parsing."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import core as bm


def load_bibtexparser(text: str) -> Tuple[List[bm.Entry], List[str]]:
    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
    except ImportError as exc:
        raise RuntimeError(
            "bibtexparser is required. Install with: pip install bibtexparser"
        ) from exc

    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = False

    bib_db = bibtexparser.loads(text, parser=parser)

    entries: List[bm.Entry] = []
    for raw in bib_db.entries:
        entry_type = (raw.get("ENTRYTYPE") or "").lower().strip()
        key = (raw.get("ID") or raw.get("id") or "").strip()
        if not entry_type or not key:
            continue
        fields: Dict[str, str] = {}
        for name, value in raw.items():
            if name in {"ENTRYTYPE", "ID"}:
                continue
            if value is None:
                continue
            fields[name.lower()] = bm.strip_outer_braces_quotes(str(value).strip())
        entries.append(bm.Entry(entry_type=entry_type, key=key, fields=fields))

    passthrough: List[str] = []
    comments = getattr(bib_db, "comments", None) or []
    for comment in comments:
        if comment is None:
            continue
        comment_text = str(comment).strip()
        if not comment_text or comment_text.strip() in {"{", "}"}:
            continue
        if comment_text.lstrip().startswith("@"):
            recovered_entries, _ = bm.parse_bibtex(comment_text)
            if recovered_entries:
                entries.extend(recovered_entries)

    return entries, passthrough


def parse_bibtex_file(path: str) -> Tuple[List[bm.Entry], List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return load_bibtexparser(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge duplicate BibTeX entries and re-key them (bibtexparser)."
    )
    parser.add_argument("input_bib", nargs="?", help="Path to input .bib file")
    parser.add_argument("output_bib", nargs="?", help="Path to output .bib file")
    parser.add_argument(
        "--report",
        help="Optional path to write conflicts report (also prints to stdout)",
    )
    parser.add_argument(
        "--conflicts-bib",
        help="Optional path to write conflicts as .bib",
    )
    parser.add_argument(
        "--apply-corrections",
        help="Optional .bib file with corrected entries to apply to input before merging",
    )
    parser.add_argument(
        "--corrected-output",
        help="Optional path to write corrected input .bib after applying corrections",
    )
    args = parser.parse_args()

    args.input_bib = bm.resolve_existing_bib_path(args.input_bib)
    if not args.output_bib:
        args.output_bib = bm.default_output_path(args.input_bib)
    if not args.report:
        args.report = bm.derive_default_path(args.input_bib, "_conflicts.txt")
    if not args.conflicts_bib:
        args.conflicts_bib = bm.derive_default_path(args.input_bib, "_conflicts.bib")

    try:
        entries, passthrough = parse_bibtex_file(args.input_bib)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    correction_lines: List[str] = []
    correction_path = bm.resolve_corrections_path(args.apply_corrections, args.input_bib)
    if args.apply_corrections and not correction_path:
        correction_lines.append(f"Corrections file not found: {args.apply_corrections}")
    if correction_path:
        corrections, _ = parse_bibtex_file(correction_path)
        correction_lines = bm.apply_corrections(entries, corrections)
        if not args.corrected_output:
            args.corrected_output = bm.derive_default_path(args.input_bib, "_corrected.bib")
        bm.write_bibtex(args.corrected_output, entries, passthrough)

    merged_entries, report_lines, conflict_clusters = bm.merge_entries(entries)
    bm.write_bibtex(args.output_bib, merged_entries, passthrough)

    if args.conflicts_bib:
        bm.write_conflicts_bib(args.conflicts_bib, conflict_clusters, entries)

    report_text = "\n".join(correction_lines + report_lines)
    print(report_text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report_text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
