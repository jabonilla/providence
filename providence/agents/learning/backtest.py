"""LEARN-BACKTEST: Backtesting Agent.

Runs offline backtests over historical data to evaluate system performance
across different time periods and regime contexts. Computes per-period
returns, Sharpe ratios, drawdowns, and win rates.

Spec Reference: Technical Spec v2.3, Phase 4 (Learning)

Classification: FROZEN — zero LLM calls. Pure computation.

Critical Rules:
  - Offline only. Never runs during live trading.
  - No lookahead bias — only uses data available at each point in time.
  - All results immutable and content-hashed.

Input: AgentContext with metadata:
  - metadata["trade_history"]: list of trade dicts with entry/exit/pnl
  - metadata["regime_history"]: list of regime dicts with timestamps
  - metadata["backtest_start"]: ISO timestamp string
  - metadata["backtest_end"]: ISO timestamp string
  - metadata["period_days"]: int (default 30, period length for sub-period analysis)

Output: BacktestOutput with per-period and aggregate metrics.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.learning import BacktestOutput, BacktestPeriod

logger = structlog.get_logger()

# Default period length for sub-period analysis
DEFAULT_PERIOD_DAYS = 30

# Annualization factor (trading days per year)
TRADING_DAYS_PER_YEAR = 252


def compute_period_metrics(
    trades: list[dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
    regime_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute metrics for a single backtest sub-period.

    Args:
        trades: Trades that closed within this period.
        period_start: Period start timestamp.
        period_end: Period end timestamp.
        regime_history: Regime states for determining dominant regime.

    Returns:
        Dict with period metrics.
    """
    returns = [float(t.get("realized_pnl_bps", 0.0)) for t in trades if isinstance(t, dict)]
    total_return = sum(returns)
    count = len(returns)

    # Sharpe ratio for this period
    sharpe = 0.0
    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.0
        sharpe = round(mean_r / std_r, 4) if std_r > 0 else 0.0

    # Max drawdown
    max_dd = 0.0
    cumulative = 0.0
    peak = 0.0
    for r in returns:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    # Win rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = round(wins / count, 4) if count > 0 else 0.0

    # Average holding days
    holding_days = [float(t.get("holding_days", 0)) for t in trades if isinstance(t, dict)]
    avg_holding = round(sum(holding_days) / len(holding_days), 1) if holding_days else 0.0

    # Dominant regime
    dominant_regime = _find_dominant_regime(period_start, period_end, regime_history)

    return {
        "period_start": period_start,
        "period_end": period_end,
        "return_bps": round(total_return, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_bps": round(max_dd, 2),
        "trade_count": count,
        "win_rate": win_rate,
        "avg_holding_days": avg_holding,
        "dominant_regime": dominant_regime,
    }


def _find_dominant_regime(
    period_start: datetime,
    period_end: datetime,
    regime_history: list[dict[str, Any]],
) -> str:
    """Find the most common regime during a period.

    Args:
        period_start: Start of period.
        period_end: End of period.
        regime_history: List of regime state dicts with timestamps.

    Returns:
        Dominant regime label, or "" if none.
    """
    regime_counts: dict[str, int] = {}

    for rh in regime_history:
        if not isinstance(rh, dict):
            continue

        ts_str = rh.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(str(ts_str))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        if period_start <= ts <= period_end:
            regime = rh.get("statistical_regime", "")
            if regime:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

    if not regime_counts:
        return ""

    return max(regime_counts, key=regime_counts.get)


def compute_profit_factor(returns: list[float]) -> float:
    """Compute profit factor: gross_profits / gross_losses.

    Args:
        returns: List of per-trade returns in bps.

    Returns:
        Profit factor (>1 is profitable), or 0.0 if no losses.
    """
    gross_profits = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r < 0))

    if gross_losses <= 0:
        return round(gross_profits, 2) if gross_profits > 0 else 0.0

    return round(gross_profits / gross_losses, 4)


class LearnBacktest(BaseAgent[BacktestOutput]):
    """Offline backtesting agent.

    Runs backtests over historical trade data to evaluate system
    performance across time periods and regime contexts.

    FROZEN: Zero LLM calls. Offline only.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="LEARN-BACKTEST",
            agent_type="learning",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> BacktestOutput:
        """Run offline backtest over historical data.

        Steps:
          1. EXTRACT trade history, regime history, backtest window
          2. PARTITION trades into sub-periods
          3. COMPUTE per-period metrics
          4. COMPUTE aggregate metrics
          5. EMIT BacktestOutput

        Args:
            context: AgentContext with trade history in metadata.

        Returns:
            BacktestOutput with per-period and aggregate results.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting offline backtest")

            trade_history = context.metadata.get("trade_history", [])
            regime_history = context.metadata.get("regime_history", [])
            bt_start_str = context.metadata.get("backtest_start", "")
            bt_end_str = context.metadata.get("backtest_end", "")
            period_days = int(context.metadata.get("period_days", DEFAULT_PERIOD_DAYS))

            if not isinstance(trade_history, list):
                trade_history = []
            if not isinstance(regime_history, list):
                regime_history = []

            # Parse backtest window
            try:
                bt_start = datetime.fromisoformat(bt_start_str) if bt_start_str else context.timestamp
            except (ValueError, TypeError):
                bt_start = context.timestamp
            try:
                bt_end = datetime.fromisoformat(bt_end_str) if bt_end_str else datetime.now(timezone.utc)
            except (ValueError, TypeError):
                bt_end = datetime.now(timezone.utc)

            if bt_start.tzinfo is None:
                bt_start = bt_start.replace(tzinfo=timezone.utc)
            if bt_end.tzinfo is None:
                bt_end = bt_end.replace(tzinfo=timezone.utc)

            # Parse trade exit timestamps
            trades_with_ts: list[tuple[datetime, dict]] = []
            for t in trade_history:
                if not isinstance(t, dict):
                    continue
                exit_str = t.get("exit_timestamp", "")
                try:
                    exit_ts = datetime.fromisoformat(str(exit_str))
                    if exit_ts.tzinfo is None:
                        exit_ts = exit_ts.replace(tzinfo=timezone.utc)
                    trades_with_ts.append((exit_ts, t))
                except (ValueError, TypeError):
                    # Trades without timestamps go to the last period
                    trades_with_ts.append((bt_end, t))

            # Generate sub-periods
            periods: list[BacktestPeriod] = []
            current_start = bt_start
            delta = timedelta(days=period_days)

            while current_start < bt_end:
                current_end = min(current_start + delta, bt_end)

                # Filter trades for this period
                period_trades = [
                    t for ts, t in trades_with_ts
                    if current_start <= ts < current_end
                ]

                if period_trades:
                    metrics = compute_period_metrics(
                        period_trades, current_start, current_end, regime_history,
                    )
                    periods.append(BacktestPeriod(**metrics))

                current_start = current_end

            # Aggregate metrics
            all_returns = [float(t.get("realized_pnl_bps", 0.0)) for t in trade_history if isinstance(t, dict)]
            total_return = sum(all_returns)
            total_trades = len(all_returns)

            # Annualized return
            total_days = (bt_end - bt_start).days
            if total_days > 0:
                annualized = total_return * (TRADING_DAYS_PER_YEAR / total_days)
            else:
                annualized = 0.0

            # Annualized Sharpe
            ann_sharpe = 0.0
            if len(all_returns) >= 2:
                mean_r = sum(all_returns) / len(all_returns)
                var_r = sum((r - mean_r) ** 2 for r in all_returns) / (len(all_returns) - 1)
                std_r = math.sqrt(var_r) if var_r > 0 else 0.0
                daily_sharpe = mean_r / std_r if std_r > 0 else 0.0
                ann_sharpe = daily_sharpe * math.sqrt(min(total_trades, TRADING_DAYS_PER_YEAR))

            # Max drawdown
            max_dd = 0.0
            cumulative = 0.0
            peak = 0.0
            for r in all_returns:
                cumulative += r
                peak = max(peak, cumulative)
                max_dd = max(max_dd, peak - cumulative)

            # Overall win rate
            wins = sum(1 for r in all_returns if r > 0)
            win_rate = round(wins / total_trades, 4) if total_trades > 0 else 0.0

            # Profit factor
            pf = compute_profit_factor(all_returns)

            output = BacktestOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                backtest_start=bt_start,
                backtest_end=bt_end,
                periods=periods,
                total_return_bps=round(total_return, 2),
                annualized_return_bps=round(annualized, 2),
                annualized_sharpe=round(ann_sharpe, 4),
                max_drawdown_bps=round(max_dd, 2),
                total_trades=total_trades,
                overall_win_rate=win_rate,
                profit_factor=pf,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Backtest complete",
                periods=len(periods),
                total_return=total_return,
                sharpe=ann_sharpe,
                trades=total_trades,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"LEARN-BACKTEST processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def get_health(self) -> HealthStatus:
        if self._error_count_24h > 10:
            status = AgentStatus.UNHEALTHY
        elif self._error_count_24h > 3:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.HEALTHY
        return HealthStatus(
            agent_id=self.agent_id,
            status=status,
            last_run=self._last_run,
            last_success=self._last_success,
            error_count_24h=self._error_count_24h,
        )
