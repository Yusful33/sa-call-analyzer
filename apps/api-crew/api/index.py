"""Vercel Python entry for the CrewAI worker — loads `main:app` with `API_SERVICE_MODE=crew`.

The Vercel project root is `apps/api-crew` only; at install time we copy `../api` into `_api_src/`
so the shared FastAPI package is present in the deployment bundle (see `vercel.json` installCommand).
"""

import os
import sys
from pathlib import Path

_API_CREW_ROOT = Path(__file__).resolve().parent.parent
_bundled = _API_CREW_ROOT / "_api_src"
_sibling = _API_CREW_ROOT.parent / "api"
_API_SRC = _bundled if _bundled.is_dir() else _sibling
os.environ.setdefault("API_SERVICE_MODE", "crew")
if not _API_SRC.is_dir():
    raise RuntimeError(
        "Shared API sources not found: expected `_api_src` (Vercel install) or sibling `../api`."
    )
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))

from main import app  # noqa: E402,F401
