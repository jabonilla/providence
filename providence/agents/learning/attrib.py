"""LEARN-ATTRIB: Performance Attribution Agent.

Computes offline performance attribution across agents, tickers, and
regime contexts. Answers: "Which agents added value? Which positions
worked? How did regime context affect performance?"

Spec Reference: Technical Spec v2.3, Phase 4 (Learning)

Classification: FROZEN — zero LLM calls. Pure computation.

Critical Rules:
  - Offline only. Never runs during live trading.
  - All data immutable and content-hashed.

Input: AgentContext with metadata:
  - metadata["closed_positions"]: list of position dicts with realized PnL
  - metadata["belief_history"]: list of historical belief dicts with outcomes
  - metadata["regime_history"]: list of regime state dicts over the period
  - metadata["evaluation_start"]: ISO timestamp string
  - metadata["evaluation_end"]: ISO timestamp string

Output: AttributionOutput with per-agent and per-ticker attribution.
"""

import math
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.learning import (
    AgentAttribution,
    AttributionOutput,
    TickerAttribution,
)

logger = structlog.get_logger()


def compute_hit_rate(beliefs: list[dict[str, Any]]) -> float:
    """Compute directional hit rate for a set of beliefs.

    A belief is a "hit" if direction matched realized outcome:
    - LONG + positive return = hit
    - SHORT + negative return = hit
    - NEUTRAL is always excluded

    Args:
        beliefs: List of belief dicts with "direction" and "realized_return_bps".

    Returns:
        Hit rate [0, 1], or 0.0 if no evaluable beliefs.
    """
    evaluable = 0
    hits = 0

    for b in beliefs:
        if not isinstance(b, dict):
            continue
        direction = b.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            continue

        realized = float(b.get("realized_return_bps", 0.0))
        evaluable += 1

        if (direction == "LONG" and realized > 0) or (direction == "SHORT" and realized < 0):
            hits += 1

    return round(hits / evaluable, 4) if evaluable > 0 else 0.0


def compute_information_ratio(returns: list[float], benchmark_returns: list[float]) -> float:
    """Compute information ratio: excess return / tracking error.

    Args:
        returns: Agent-attributed returns (bps).
        benchmark_returns: Benchmark returns (bps).

    Returns:
        Information ratio, or 0.0 if insufficient data.
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    min_len = min(len(returns), len(benchmark_returns))
    excess = [returns[i] - benchmark_returns[i] for i in range(min_len)]

    avg_excess = sum(excess) / len(excess)
    variance = sum((e - avg_excess) ** 2 for e in excess) / (len(excess) - 1)
    tracking_error = math.sqrt(variance) if variance > 0 else 0.0

    return round(avg_excess / tracking_error, 4) if tracking_error > 0 else 0.0


def compute_sharpe_contribution(
    agent_returns: list[float],
    portfolio_returns: list[float],
    weight: float,
) -> float:
    """Compute agent's contribution to portfolio Sharpe ratio.

    Simplified: weight * agent_sharpe, where agent_sharpe is
    mean(agent_returns) / std(agent_returns).

    Args:
        agent_returns: Per-period returns attributed to this agent (bps).
        portfolio_returns: Total portfolio per-period returns (bps).
        weight: Agent's average portfolio weight contribution.

    Returns:
        Sharpe contribution.
    """
    if len(agent_returns) < 2:
        return 0.0

    mean_ret = sum(agent_returns) / len(agent_returns)
    variance = sum((r - mean_ret) ** 2 for r in agent_returns) / (len(agent_returns) - 1)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0

    agent_sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    return round(weight * agent_sharpe, 4)


def attribute_agent(
    agent_id: str,
    beliefs: list[dict[str, Any]],
    benchmark_returns: list[float],
    portfolio_returns: list[float],
) -> dict[str, Any]:
    """Compute full attribution for a single agent.

    Args:
        agent_id: Agent identifier.
        beliefs: Beliefs produced by this agent with outcomes.
        benchmark_returns: Benchmark returns for IR calculation.
        portfolio_returns: Portfolio returns for Sharpe contribution.

    Returns:
        Dict with attribution fields.
    """
    total_produced = len(beliefs)
    acted_on = [b for b in beliefs if b.get("was_acted_on", False)]
    acted_count = len(acted_on)

    hit_rate = compute_hit_rate(acted_on)

    confidences = [float(b.get("raw_confidence", 0.0)) for b in beliefs if isinstance(b, dict)]
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    returns = [float(b.get("realized_return_bps", 0.0)) for b in acted_on if isinstance(b, dict)]
    avg_return = round(sum(returns) / len(returns), 2) if returns else 0.0

    ir = compute_information_ratio(returns, benchmark_returns)

    weight = acted_count / max(len(portfolio_returns), 1)
    sharpe_contrib = compute_sharpe_contribution(returns, portfolio_returns, weight)

    max_dd = 0.0
    if returns:
        cumulative = 0.0
        peak = 0.0
        for r in returns:
            cumulative += r
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

    value_added = avg_return - (sum(benchmark_returns) / len(benchmark_returns) if benchmark_returns else 0.0)

    return {
        "agent_id": agent_id,
        "total_beliefs_produced": total_produced,
        "beliefs_acted_on": acted_count,
        "hit_rate": hit_rate,
        "avg_confidence": avg_confidence,
        "avg_return_bps": avg_return,
        "sharpe_contribution": sharpe_contrib,
        "information_ratio": ir,
        "max_drawdown_contribution_bps": round(max_dd, 2),
        "value_added_bps": round(value_added, 2),
    }


def attribute_ticker(
    ticker: str,
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute attribution for a single ticker.

    Args:
        ticker: Ticker symbol.
        positions: Closed positions for this ticker.

    Returns:
        Dict with ticker attribution fields.
    """
    total_pnl = sum(float(p.get("realized_pnl_bps", 0.0)) for p in positions)
    total_days = sum(int(p.get("holding_days", 0)) for p in positions)
    weights = [float(p.get("avg_weight", 0.0)) for p in positions]
    avg_weight = sum(weights) / len(weights) if weights else 0.0

    agents: set[str] = set()
    for p in positions:
        for a in p.get("contributing_agents", []):
            if isinstance(a, str):
                agents.add(a)

    last_exit = positions[-1] if positions else {}
    exit_reason = last_exit.get("exit_reason", "")
    regime = last_exit.get("regime_during_hold", "")

    return {
        "ticker": ticker,
        "total_pnl_bps": round(total_pnl, 2),
        "holding_days": total_days,
        "avg_weight": round(avg_weight, 4),
        "contributing_agents": sorted(agents),
        "exit_reason": exit_reason,
        "regime_during_hold": regime,
    }


class LearnAttrib(BaseAgent[AttributionOutput]):
    """Offline performance attribution agent.

    Computes per-agent and per-ticker attribution over a historical
    evaluation window. Answers who added value and how.

    FROZEN: Zero LLM calls. Offline only.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="LEARN-ATTRIB",
            agent_type="learning",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> AttributionOutput:
        """Run offline performance attribution.

        Steps:
          1. EXTRACT closed positions, belief history, regime history
          2. GROUP beliefs by agent_id
          3. COMPUTE per-agent attribution
          4. GROUP positions by ticker
          5. COMPUTE per-ticker attribution
          6. COMPUTE portfolio-level metrics
          7. EMIT AttributionOutput

        Args:
            context: AgentContext with historical data in metadata.

        Returns:
            AttributionOutput with full attribution breakdown.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting offline attribution")

            closed_positions = context.metadata.get("closed_positions", [])
            belief_history = context.metadata.get("belief_history", [])
            regime_history = context.metadata.get("regime_history", [])
            eval_start_str = context.metadata.get("evaluation_start", "")
            eval_end_str = context.metadata.get("evaluation_end", "")

            if not isinstance(closed_positions, list):
                closed_positions = []
            if not isinstance(belief_history, list):
                belief_history = []

            # Parse evaluation window
            try:
                eval_start = datetime.fromisoformat(eval_start_str) if eval_start_str else context.timestamp
            except (ValueError, TypeError):
                eval_start = context.timestamp
            try:
                eval_end = datetime.fromisoformat(eval_end_str) if eval_end_str else datetime.now(timezone.utc)
            except (ValueError, TypeError):
                eval_end = datetime.now(timezone.utc)

            if eval_start.tzinfo is None:
                eval_start = eval_start.replace(tzinfo=timezone.utc)
            if eval_end.tzinfo is None:
                eval_end = eval_end.replace(tzinfo=timezone.utc)

            # Compute benchmark returns (equal-weight average of all position returns)
            all_returns = [
                float(p.get("realized_pnl_bps", 0.0))
                for p in closed_positions
                if isinstance(p, dict)
            ]
            benchmark_returns = all_returns if all_returns else [0.0]

            # Group beliefs by agent
            beliefs_by_agent: dict[str, list[dict]] = {}
            for b in belief_history:
                if not isinstance(b, dict):
                    continue
                aid = b.get("agent_id", "")
                if aid:
                    beliefs_by_agent.setdefault(aid, []).append(b)

            # Per-agent attribution
            agent_attrs: list[AgentAttribution] = []
            for aid, beliefs in sorted(beliefs_by_agent.items()):
                attr_dict = attribute_agent(aid, beliefs, benchmark_returns, all_returns)
                agent_attrs.append(AgentAttribution(**attr_dict))

            # Group positions by ticker
            positions_by_ticker: dict[str, list[dict]] = {}
            for p in closed_positions:
                if not isinstance(p, dict):
                    continue
                ticker = p.get("ticker", "")
                if ticker:
                    positions_by_ticker.setdefault(ticker, []).append(p)

            # Per-ticker attribution
            ticker_attrs: list[TickerAttribution] = []
            for ticker, positions in sorted(positions_by_ticker.items()):
                attr_dict = attribute_ticker(ticker, positions)
                ticker_attrs.append(TickerAttribution(**attr_dict))

            # Portfolio-level metrics
            portfolio_return = sum(all_returns)
            portfolio_sharpe = 0.0
            if len(all_returns) >= 2:
                mean_r = sum(all_returns) / len(all_returns)
                var_r = sum((r - mean_r) ** 2 for r in all_returns) / (len(all_returns) - 1)
                std_r = math.sqrt(var_r) if var_r > 0 else 0.0
                portfolio_sharpe = round(mean_r / std_r, 4) if std_r > 0 else 0.0

            max_dd = 0.0
            cumulative = 0.0
            peak = 0.0
            for r in all_returns:
                cumulative += r
                peak = max(peak, cumulative)
                max_dd = max(max_dd, peak - cumulative)

            output = AttributionOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                evaluation_start=eval_start,
                evaluation_end=eval_end,
                agent_attributions=agent_attrs,
                ticker_attributions=ticker_attrs,
                portfolio_sharpe=round(portfolio_sharpe, 4),
                portfolio_return_bps=round(portfolio_return, 2),
                portfolio_max_drawdown_bps=round(max_dd, 2),
                total_trades=len(closed_positions),
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Attribution complete",
                agents=len(agent_attrs),
                tickers=len(ticker_attrs),
                portfolio_return=portfolio_return,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"LEARN-ATTRIB processing failed: {e}",
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
