#!/usr/bin/env python3
"""Post-pip prune for Vercel Python (250 MiB unzipped serverless cap).

CrewAI depends on ``uv`` (CLI) and PyPI metadata pulls ``kubernetes`` for Chroma;
runtime imports do not need the ``uv`` binaries or the ``kubernetes`` client tree.
``sympy`` is pulled by some ``onnxruntime`` builds and is not required at inference.

Set ``PRUNE_VERCEL_CREW_WORKER=1`` (``id-pain-api-crew`` install) for extra trims: ONNX
training/tooling trees, BigQuery client leftovers (crew workers skip BQ).
"""
from __future__ import annotations

import os
import shutil
import site
import sys
from pathlib import Path


def _rm(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        print(f"prune: removed file {path}")
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"prune: removed dir {path}")


def prune_crew_worker_extras(root: Path) -> None:
    """Extra removals for ``id-pain-api-crew`` only (set ``PRUNE_VERCEL_CREW_WORKER=1``).

    - ``onnxruntime/{transformers,quantization,tools}``: not needed for Chroma default embeddings / inference.
    - ``google/cloud/bigquery``: app never imports BQ when ``API_SERVICE_MODE=crew``; ``pip uninstall`` may leave crumbs.
    - Every ``tests`` / ``__pycache__`` directory under this site root (often tens of MB across NumPy, gRPC, etc.).
    """
    ort = root / "onnxruntime"
    if ort.is_dir():
        for sub in ("transformers", "quantization", "tools"):
            _rm(ort / sub)
    bq = root / "google" / "cloud" / "bigquery"
    _rm(bq)
    for meta in root.glob("google_cloud_bigquery-*.dist-info"):
        _rm(meta)
    # Delete deepest paths first so os.walk does not descend into removed trees.
    hits: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        p = Path(dirpath)
        if p.name in ("tests", "__pycache__") and p.is_dir():
            hits.append(p)
    for p in sorted(hits, key=lambda x: len(x.parts), reverse=True):
        _rm(p)


def prune_root(root: Path) -> None:
    if not root.is_dir():
        return
    bin_dir = root / "bin"
    if bin_dir.is_dir():
        for name in ("uv", "uvx"):
            p = bin_dir / name
            if p.is_file():
                _rm(p)
    _rm(root / "kubernetes")
    for meta in root.glob("kubernetes-*.dist-info"):
        _rm(meta)
    _rm(root / "sympy")
    for meta in root.glob("sympy-*.dist-info"):
        _rm(meta)
    _rm(root / "mpmath")
    for meta in root.glob("mpmath-*.dist-info"):
        _rm(meta)
    _rm(root / "pre_commit")
    for meta in root.glob("pre_commit-*.dist-info"):
        _rm(meta)
    _rm(root / "virtualenv")
    for meta in root.glob("virtualenv-*.dist-info"):
        _rm(meta)
    _rm(root / "nodeenv.py")
    for meta in root.glob("nodeenv-*.dist-info"):
        _rm(meta)
    if os.environ.get("PRUNE_VERCEL_CREW_WORKER", "").strip().lower() in ("1", "true", "yes"):
        prune_crew_worker_extras(root)


def _candidate_roots() -> list[Path]:
    roots: set[Path] = set()
    for p in site.getsitepackages():
        if p:
            roots.add(Path(p).resolve())
    sp = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if sp.is_dir():
        roots.add(sp.resolve())
    for part in os.environ.get("PRUNE_SITE_PACKAGES", "").split(os.pathsep):
        part = part.strip()
        if part:
            roots.add(Path(part).resolve())
    # ``pip install -t <dir>`` layouts: that dir is on PYTHONPATH but not always in getsitepackages().
    for p in sys.path:
        if not p or p in (".", ""):
            continue
        path = Path(p).resolve()
        if (path / "crewai").is_dir():
            roots.add(path)
    return sorted(roots)


def main() -> int:
    seen: set[str] = set()
    for root in _candidate_roots():
        key = str(root)
        if not root.is_dir() or key in seen:
            continue
        seen.add(key)
        prune_root(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
