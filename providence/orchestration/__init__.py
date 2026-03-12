"""Providence orchestration layer.

Wires all 35 agents together across 4 pipeline loops:
  - Main: Cognition → Regime → Decision → Execution
  - Exit: COGNIT-EXIT → INVALID-MON → THESIS-RENEW → SHADOW-EXIT → RENEW-MON
  - Learning: LEARN-ATTRIB → LEARN-CALIB → LEARN-RETRAIN → LEARN-BACKTEST
  - Governance: GOVERN-CAPITAL → GOVERN-MATURITY → GOVERN-OVERSIGHT → GOVERN-POLICY

Lazy imports to avoid circular dependency:
  run_store → orchestration.models → (this __init__) → runner → run_store
"""


def __getattr__(name: str):
    """Lazy-load orchestration submodules on first attribute access."""
    if name in ("PipelineRun", "RunStatus", "StageResult", "StageStatus"):
        from providence.orchestration.models import PipelineRun, RunStatus, StageResult, StageStatus
        return {"PipelineRun": PipelineRun, "RunStatus": RunStatus,
                "StageResult": StageResult, "StageStatus": StageStatus}[name]
    if name == "Orchestrator":
        from providence.orchestration.orchestrator import Orchestrator
        return Orchestrator
    if name == "ProvidenceRunner":
        from providence.orchestration.runner import ProvidenceRunner
        return ProvidenceRunner
    if name == "PipelineStage":
        from providence.orchestration.stage import PipelineStage
        return PipelineStage
    raise AttributeError(f"module 'providence.orchestration' has no attribute {name!r}")


__all__ = [
    "Orchestrator",
    "PipelineRun",
    "PipelineStage",
    "ProvidenceRunner",
    "RunStatus",
    "StageResult",
    "StageStatus",
]
