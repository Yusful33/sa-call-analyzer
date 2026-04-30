"""Vercel Python entry point — re-exports the FastAPI app from ../main.py.

Vercel detects this `api/` folder, runs `main.py`'s `app` as an ASGI Function
behind the rewrite in `vercel.json`. Cold starts include observability + LangChain
imports, so expect ~5-10s on the first request after a long idle.
"""
import sys
from pathlib import Path

# Ensure apps/api is importable so `import main` works
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402,F401  (Vercel discovers `app` ASGI handler)
