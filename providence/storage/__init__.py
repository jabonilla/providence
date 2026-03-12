"""Storage layer — persistence for fragments, beliefs, and pipeline runs.

All stores are append-only (immutable data), matching Providence's
core invariant that data is never modified after creation.

Lazy imports to avoid circular dependency:
  storage.__init__ → run_store → orchestration.models → orchestration.__init__
  → runner → storage.run_store (circular)
"""


def __getattr__(name: str):
    """Lazy-load store classes on first attribute access."""
    if name == "FragmentStore":
        from providence.storage.fragment_store import FragmentStore
        return FragmentStore
    if name == "BeliefStore":
        from providence.storage.belief_store import BeliefStore
        return BeliefStore
    if name == "RunStore":
        from providence.storage.run_store import RunStore
        return RunStore
    raise AttributeError(f"module 'providence.storage' has no attribute {name!r}")


__all__ = [
    "FragmentStore",
    "BeliefStore",
    "RunStore",
]
