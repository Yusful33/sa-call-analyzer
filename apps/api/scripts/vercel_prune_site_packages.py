#!/usr/bin/env python3
"""Post-pip prune for Vercel Python (250 MiB unzipped serverless cap).

CrewAI depends on the ``uv`` PyPI package (ships a **~50 MiB** ``bin/uv`` under
``sys.prefix/bin``, not ``site-packages/bin``) and pulls ``kubernetes`` for Chroma;
runtime imports do not need those CLIs or the ``kubernetes`` client tree.
The ``ty`` package also drops a large ``bin/ty`` next to ``uv``.
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


# Heavy packages that should NEVER be in the Vercel bundle
# These are transitive deps from ChromaDB that we don't need (using API-based LLMs)
FORCE_REMOVE_PACKAGES = [
    # ChromaDB and vector DB deps
    "chromadb",
    "onnxruntime",
    "flatbuffers",
    "mmh3",
    "pypdfium2",
    # HuggingFace ecosystem
    "huggingface_hub",
    "huggingface-hub",
    "tokenizers",
    "sentence_transformers",
    "sentence-transformers",
    "hf_xet",
    "hf-xet",
    # ML frameworks (should never be pulled but just in case)
    "torch",
    "tensorflow",
    "transformers",
]


def prune_force_remove_packages(root: Path) -> None:
    """Force-remove heavy packages that should never be in the Vercel bundle.
    
    These are mostly ChromaDB transitive dependencies that we don't need
    because we use API-based LLMs, not local embeddings.
    """
    for pkg in FORCE_REMOVE_PACKAGES:
        # Try both underscore and hyphen variants
        for variant in (pkg, pkg.replace("_", "-"), pkg.replace("-", "_")):
            pkg_dir = root / variant
            _rm(pkg_dir)
            # Also remove dist-info directories
            for meta in root.glob(f"{variant}*.dist-info"):
                _rm(meta)
            for meta in root.glob(f"{variant.replace('-', '_')}*.dist-info"):
                _rm(meta)


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


def prune_tests_and_cache(root: Path) -> None:
    """Remove tests and __pycache__ directories to save space."""
    hits: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        p = Path(dirpath)
        if p.name in ("tests", "test", "__pycache__", ".pytest_cache") and p.is_dir():
            hits.append(p)
    # Delete deepest paths first so os.walk does not descend into removed trees.
    for p in sorted(hits, key=lambda x: len(x.parts), reverse=True):
        _rm(p)


def prune_root(root: Path) -> None:
    if not root.is_dir():
        return
    
    # Always force-remove heavy packages (ChromaDB chain, etc.)
    prune_force_remove_packages(root)
    
    # Remove CLI binaries
    bin_dir = root / "bin"
    if bin_dir.is_dir():
        for name in ("uv", "uvx", "ty"):
            p = bin_dir / name
            if p.is_file() or p.is_symlink():
                _rm(p)
    
    # Remove dev/build packages
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
    
    # Always prune tests and cache directories for size reduction
    prune_tests_and_cache(root)
    
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


def prune_sys_prefix_bin() -> None:
    """Remove large CLIs from ``sys.prefix/bin`` (Vercel bundles this tree for Python).

    ``uv`` / ``ty`` wheels install executables here; ``prune_root`` only touched
    ``site-packages/*/bin`` and missed them, which bloated the function to >250 MiB.
    """
    prefix = Path(sys.prefix).resolve()
    bin_dir = prefix / "bin"
    if not bin_dir.is_dir():
        return
    for name in ("uv", "uvx", "ty"):
        p = bin_dir / name
        if p.is_file() or p.is_symlink():
            _rm(p)


def main() -> int:
    prune_sys_prefix_bin()
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
