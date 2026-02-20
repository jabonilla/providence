"""Providence orchestration layer.

Wires all 35 agents together across 4 pipeline loops:
  - Main: Cognition → Regime → Decision → Execution
  - Exit: COGNIT-EXIT → INVALID-MON → THESIS-RENEW → SHADOW-EXIT → RENEW-MON
  - Learning: LEARN-ATTRIB → LEARN-CALIB → LEARN-RETRAIN → LEARN-BACKTEST
  - Governance: GOVERN-CAPITAL → GOVERN-MATURITY → GOVERN-OVERSIGHT → GOVERN-POLICY
"""

from providence.orchestration.models import PipelineRun, RunStatus, StageResult, StageStatus
from providence.orchestration.orchestrator import Orchestrator
from providence.orchestration.runner import ProvidenceRunner
from providence.orchestration.stage import PipelineStage

__all__ = [
    "Orchestrator",
    "PipelineRun",
    "PipelineStage",
    "ProvidenceRunner",
    "RunStatus",
    "StageResult",
    "StageStatus",
]
