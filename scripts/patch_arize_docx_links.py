#!/usr/bin/env python3
"""
Rewrite Arize doc hyperlinks inside PoC/PoT .docx templates to canonical docs.arize.com URLs.

Run from repo root:
  python scripts/patch_arize_docx_links.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent

# Substrings replaced inside document XML parts (preserves OOXML validity)
PLAIN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("https://arize.com/docs/ax/", "https://docs.arize.com/ax/"),
    ("http://arize.com/docs/ax/", "https://docs.arize.com/ax/"),
)

# Paths after .../docs/ax/ → canonical AX paths under docs.arize.com/ax/
PATH_SUFFIX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("agents/arize-skills", "set-up-with-ai-assistants"),
    ("agents/tracing-assistant", "set-up-with-ai-assistants"),
    ("alyx/meet-alyx", "alyx"),
    ("evaluate/evaluators", "evaluate/create-evaluators"),
)


def patch_xml_text(data: bytes) -> bytes:
    text = data.decode("utf-8")
    for a, b in PLAIN_REPLACEMENTS:
        text = text.replace(a, b)
    for suffix_from, suffix_to in PATH_SUFFIX_REPLACEMENTS:
        text = text.replace(
            f"https://docs.arize.com/ax/{suffix_from}",
            f"https://docs.arize.com/ax/{suffix_to}",
        )
    return text.encode("utf-8")


def patch_docx(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as td:
        work = Path(td) / "w"
        with zipfile.ZipFile(src, "r") as zin:
            zin.extractall(work)
        for rel in ["word/document.xml", "word/header1.xml", "word/footer1.xml"]:
            path = work / rel
            if path.is_file():
                path.write_bytes(patch_xml_text(path.read_bytes()))
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for fp in sorted(work.rglob("*")):
                if fp.is_file():
                    arc = fp.relative_to(work).as_posix()
                    zout.write(fp, arcname=arc)


def main() -> int:
    src_dir = ROOT / "templates" / "poc_pot"
    if not src_dir.is_dir():
        print(f"Missing {src_dir}", file=sys.stderr)
        return 1

    destinations = [
        ROOT / "templates" / "poc_pot",
        ROOT / "apps" / "api" / "templates" / "poc_pot",
        ROOT / "apps" / "api" / "api" / "templates" / "poc_pot",
    ]

    for name in ("poc_saas", "poc_vpc", "pot"):
        doc = src_dir / f"{name}.docx"
        if not doc.is_file():
            continue
        for dest_dir in destinations:
            out = dest_dir / f"{name}.docx"
            patch_docx(doc, out)
            print(f"Patched {name}.docx -> {out}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
