"""Providence REST API — FastAPI application.

Exposes pipeline execution, health monitoring, agent status,
storage queries, and portfolio management over HTTP.
"""

def create_app(*args, **kwargs):
    """Lazy import to avoid pulling in fastapi at package-import time."""
    from providence.api.app import create_app as _create_app
    return _create_app(*args, **kwargs)

__all__ = ["create_app"]
