#!/usr/bin/env python3
"""CI/local check: canonical docs.arize.com URLs must return HTTP OK.

Uses only `httpx` (same dependency as FastAPI bundle). Exit 1 if any link fails."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arize_doc_links import validate_doc_links_sync  # noqa: E402


def main() -> int:
    result = validate_doc_links_sync(timeout_seconds=20.0)
    broken = result.get("broken") or []
    verbose_json = (
        os.environ.get("DOC_LINKS_JSON", "").strip().lower() in ("1", "true", "yes")
        or not os.environ.get("GITHUB_ACTIONS", "").strip()
    )
    if verbose_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") != "healthy":
        print(
            f"FAILED: {len(broken)} broken doc link(s): {', '.join(broken)}",
            file=sys.stderr,
        )
        if not verbose_json:
            print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    msg = (
        f"OK: all {result.get('total', 0)} canonical doc links reachable"
        if verbose_json
        else f"Doc links OK ({result.get('total', 0)} checked)"
    )
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
