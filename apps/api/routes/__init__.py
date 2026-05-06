"""
FastAPI route modules.

This package contains route handlers extracted from main.py for better organization.
"""

from .health import router as health_router
from .prospect import router as prospect_router

__all__ = ["health_router", "prospect_router"]
