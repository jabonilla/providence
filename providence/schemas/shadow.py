"""Shadow mode schemas — signal recording and simulated execution.

These schemas capture what the pipeline *would* have done without
actually submitting orders to a broker. Used during Shadow Mode
(Launch Plan Phase B) to validate signal quality before paper trading.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from providence.schemas.enums import Action, Direction, SystemMode


class ShadowSignal(BaseModel, frozen=True):
    """A recorded trading signal from a shadow mode pipeline run.

    Captures everything the pipeline decided to do so it can be
    compared against actual market outcomes later.
    """
    signal_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str
    action: Action
    direction: Direction
    target_weight: float
    confidence: float
    approved: bool  # Did EXEC-VALIDATE approve it?
    rejection_reasons: list[str] = Field(default_factory=list)
    adjusted_weight: float = 0.0
    risk_mode_applied: str = "NORMAL"

    # Simulated execution fields (filled by ShadowExecutionService)
    simulated_entry_price: Optional[float] = None
    simulated_fill_qty: Optional[int] = None
    simulated_notional: Optional[float] = None

    # For later comparison against actual market outcomes
    price_at_signal: Optional[float] = None
    price_1d_later: Optional[float] = None
    price_5d_later: Optional[float] = None
    price_20d_later: Optional[float] = None
    realized_return_1d: Optional[float] = None
    realized_return_5d: Optional[float] = None
    realized_return_20d: Optional[float] = None


class ShadowRunSummary(BaseModel, frozen=True):
    """Summary of a single shadow mode pipeline cycle.

    Aggregates all signals from one pipeline run for reporting.
    """
    run_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_mode: SystemMode = SystemMode.SHADOW
    signals: list[ShadowSignal] = Field(default_factory=list)
    total_signals: int = 0
    approved_signals: int = 0
    rejected_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    risk_mode: str = "NORMAL"
    regime_state: str = "LOW_VOL_TRENDING"


class ShadowPerformanceReport(BaseModel, frozen=True):
    """Aggregated performance report across multiple shadow runs.

    Computed by the shadow report generator to evaluate signal quality
    before advancing to paper trading (Launch Plan Phase C criteria).
    """
    report_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    total_runs: int = 0
    total_signals: int = 0
    total_approved: int = 0

    # Directional accuracy (signals where direction matched market move)
    accuracy_1d: Optional[float] = None  # % of signals correct at 1 day
    accuracy_5d: Optional[float] = None  # % of signals correct at 5 days
    accuracy_20d: Optional[float] = None  # % of signals correct at 20 days

    # Hypothetical P&L (if signals were executed)
    hypothetical_return_1d: Optional[float] = None
    hypothetical_return_5d: Optional[float] = None
    hypothetical_return_20d: Optional[float] = None

    # Signal quality metrics
    avg_confidence: Optional[float] = None
    confidence_calibration: Optional[float] = None  # Brier-like score
    long_short_ratio: Optional[float] = None
    avg_signals_per_run: Optional[float] = None

    # Phase B success criteria (from Launch Plan)
    meets_accuracy_threshold: bool = False    # > 55% directional accuracy
    meets_sharpe_threshold: bool = False      # > 0.5 hypothetical Sharpe
    meets_stability_threshold: bool = False   # No catastrophic draws
    ready_for_paper_trading: bool = False     # All criteria met
