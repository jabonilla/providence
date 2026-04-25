"""ChatEngine — intent classification and response generation for the chat API.

v1 implementation: pure keyword matching + store queries + template formatting.
No LLM calls. Fast and free.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providence.api.deps import AppState

logger = structlog.get_logger()


class Intent(str, Enum):
    """Chat intent categories."""

    PORTFOLIO = "PORTFOLIO"
    POSITIONS = "POSITIONS"
    BELIEFS = "BELIEFS"
    REGIME = "REGIME"
    AGENTS = "AGENTS"
    PIPELINE = "PIPELINE"
    SHADOW = "SHADOW"
    GENERAL = "GENERAL"


# Keyword patterns for each intent (checked in order, first match wins)
_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (
        Intent.POSITIONS,
        [
            r"\bposition[s]?\b",
            r"\b[A-Z]{1,5}\b(?=.*(?:price|entry|pnl|p&l|weight|held|hold))",
        ],
    ),
    (
        Intent.PORTFOLIO,
        [
            r"\bportfolio\b",
            r"\bholding[s]?\b",
            r"\bequity\b",
            r"\bcash\b",
            r"\bexposure\b",
            r"\bp&l\b",
            r"\bpnl\b",
            r"\bvalue\b",
            r"\bnet worth\b",
            r"\bdrawdown\b",
        ],
    ),
    (
        Intent.BELIEFS,
        [
            r"\bbelief[s]?\b",
            r"\bthes[ie]s\b",
            r"\bconviction[s]?\b",
            r"\bwhat do agents think\b",
            r"\banalysis\b",
            r"\bbullish\b",
            r"\bbearish\b",
        ],
    ),
    (
        Intent.REGIME,
        [
            r"\bregime\b",
            r"\brisk mode\b",
            r"\brisk level\b",
            r"\bvolatility\b",
            r"\bmarket state\b",
            r"\bhmm\b",
            r"\bmarket condition\b",
        ],
    ),
    (
        Intent.AGENTS,
        [
            r"\bagent[s]?\b",
            r"\bhealth\b",
            r"\bstatus\b",
            r"\bsubsystem[s]?\b",
        ],
    ),
    (
        Intent.PIPELINE,
        [
            r"\bpipeline\b",
            r"\brun[s]?\b",
            r"\bcycle[s]?\b",
            r"\bexecution\b",
            r"\bsuccess rate\b",
            r"\bstage[s]?\b",
        ],
    ),
    (
        Intent.SHADOW,
        [
            r"\bshadow\b",
            r"\bsignal[s]?\b",
            r"\bsimulat\w*\b",
            r"\bpaper trad\w*\b",
            r"\bbackfill\b",
            r"\bphase b\b",
        ],
    ),
]

# Regex to extract tickers from messages
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Known tickers to avoid matching generic words
_COMMON_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META",
    "JPM", "BAC", "WFC", "GS", "BRK", "V", "MA", "UNH", "JNJ", "PG",
    "XOM", "CVX", "HD", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CRM",
    "COST", "WMT", "TGT", "LOW", "SBUX", "NKE", "MCD", "PEP", "KO",
    "ABBV", "MRK", "PFE", "LLY", "TMO", "ABT", "DHR", "BMY", "GILD",
    "SPY", "QQQ", "IWM", "VTI", "VOO",
}


def _extract_tickers(text: str) -> list[str]:
    """Extract plausible stock tickers from a message."""
    candidates = _TICKER_RE.findall(text)
    # Filter to known tickers to avoid matching words like "THE", "AND", etc.
    return [t for t in candidates if t in _COMMON_TICKERS]


class ChatEngine:
    """Intent classifier + response generator for the chat interface.

    Queries AppState stores based on classified intent and formats
    natural language responses with structured citations.
    """

    def __init__(self, state: AppState) -> None:
        self._state = state

    def classify_intent(self, message: str) -> Intent:
        """Classify a user message into an intent category."""
        text = message.lower()
        for intent, patterns in _INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        # Check for bare tickers as a positions query
        tickers = _extract_tickers(message)
        if tickers:
            return Intent.POSITIONS
        return Intent.GENERAL

    def process(
        self,
        message: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Process a user message and return (response_text, citations).

        Args:
            message: The user's message text.
            conversation_history: Optional list of prior messages in the
                conversation (each dict with 'role', 'content', 'timestamp').
                Stored for future LLM-based responses but not yet used in
                intent classification.

        Returns:
            Tuple of (natural language response, list of citation dicts)
        """
        intent = self.classify_intent(message)
        logger.debug(
            "Chat intent classified",
            intent=intent.value,
            message=message[:80],
            history_length=len(conversation_history) if conversation_history else 0,
        )

        handler = {
            Intent.PORTFOLIO: self._handle_portfolio,
            Intent.POSITIONS: self._handle_positions,
            Intent.BELIEFS: self._handle_beliefs,
            Intent.REGIME: self._handle_regime,
            Intent.AGENTS: self._handle_agents,
            Intent.PIPELINE: self._handle_pipeline,
            Intent.SHADOW: self._handle_shadow,
            Intent.GENERAL: self._handle_general,
        }[intent]

        try:
            return handler(message)
        except Exception as exc:
            logger.error("Chat handler error", intent=intent.value, error=str(exc))
            return (
                "I encountered an error processing your request. Please try again.",
                [],
            )

    # ------------------------------------------------------------------
    # Intent Handlers
    # ------------------------------------------------------------------

    def _handle_portfolio(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle portfolio/holdings queries."""
        tracker = self._state.portfolio_tracker
        if tracker is None:
            return ("Portfolio tracking is not initialized.", [])

        positions = tracker.positions
        equity = tracker.equity
        citations: list[dict[str, Any]] = []

        if not positions:
            return (
                f"The portfolio is currently empty with ${equity:,.2f} in equity.",
                [],
            )

        # Build snapshot-like data without persisting a new snapshot
        total_unrealized = sum(p.unrealized_pnl for p in positions.values())
        total_realized = sum(p.realized_pnl for p in positions.values())
        long_count = sum(1 for p in positions.values() if p.quantity > 0)
        short_count = sum(1 for p in positions.values() if p.quantity < 0)

        # Top positions by absolute weight
        sorted_positions = sorted(
            positions.values(), key=lambda p: abs(p.weight), reverse=True
        )
        top_5 = sorted_positions[:5]

        lines = [
            f"**Portfolio Overview**",
            f"- Total equity: ${equity:,.2f}",
            f"- Positions: {len(positions)} ({long_count} long, {short_count} short)",
            f"- Unrealized P&L: ${total_unrealized:,.2f}",
            f"- Realized P&L: ${total_realized:,.2f}",
            f"- Drawdown: {tracker.drawdown_pct:.1%}",
            "",
            "**Top Positions by Weight:**",
        ]

        for pos in top_5:
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            lines.append(
                f"- {pos.ticker}: {pos.weight:.1%} weight, "
                f"${pos.current_price:,.2f} ({pnl_sign}${pos.unrealized_pnl:,.2f})"
            )
            citations.append({
                "type": "position",
                "id": pos.ticker,
                "label": f"{pos.ticker} position",
                "url": f"/api/v1/portfolio/positions/{pos.ticker}",
            })

        return ("\n".join(lines), citations)

    def _handle_positions(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle specific position queries."""
        tracker = self._state.portfolio_tracker
        if tracker is None:
            return ("Portfolio tracking is not initialized.", [])

        tickers = _extract_tickers(message)
        citations: list[dict[str, Any]] = []

        if tickers:
            # Query specific tickers
            lines: list[str] = []
            for ticker in tickers:
                pos = tracker.get_position(ticker)
                if pos:
                    pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                    lines.extend([
                        f"**{ticker}** ({pos.side.value})",
                        f"- Quantity: {pos.quantity}",
                        f"- Entry: ${pos.avg_entry_price:,.2f} | Current: ${pos.current_price:,.2f}",
                        f"- Unrealized P&L: {pnl_sign}${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_pct:+.1%})",
                        f"- Weight: {pos.weight:.1%} | Sector: {pos.sector}",
                        f"- Days held: {pos.days_held}",
                        "",
                    ])
                    citations.append({
                        "type": "position",
                        "id": ticker,
                        "label": f"{ticker} position",
                        "url": f"/api/v1/portfolio/positions/{ticker}",
                    })
                else:
                    lines.append(f"No open position in {ticker}.")
                    lines.append("")
            return ("\n".join(lines).strip(), citations)

        # No specific ticker — list all positions
        positions = tracker.positions
        if not positions:
            return ("No open positions.", [])

        lines = [f"**All Open Positions ({len(positions)}):**", ""]
        for ticker, pos in sorted(positions.items()):
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            lines.append(
                f"- **{ticker}** ({pos.side.value}): {pos.weight:.1%} weight, "
                f"{pnl_sign}${pos.unrealized_pnl:,.2f} P&L"
            )
            citations.append({
                "type": "position",
                "id": ticker,
                "label": f"{ticker} position",
                "url": f"/api/v1/portfolio/positions/{ticker}",
            })

        return ("\n".join(lines), citations)

    def _handle_beliefs(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle belief/thesis queries."""
        belief_store = self._state.belief_store
        citations: list[dict[str, Any]] = []
        tickers = _extract_tickers(message)

        if tickers:
            # Beliefs for specific tickers
            lines: list[str] = []
            for ticker in tickers:
                beliefs = belief_store.get_latest_by_ticker(ticker)
                if beliefs:
                    lines.append(f"**Latest beliefs for {ticker}:**")
                    for bo in beliefs:
                        for b in bo.beliefs:
                            if b.ticker == ticker:
                                conf_pct = f"{b.raw_confidence:.0%}"
                                lines.append(
                                    f"- {bo.agent_id}: {b.direction.value} "
                                    f"({b.magnitude.value}), {conf_pct} confidence"
                                )
                                if b.thesis_summary:
                                    lines.append(f"  {b.thesis_summary[:120]}")
                        citations.append({
                            "type": "belief",
                            "id": str(bo.belief_id),
                            "label": f"{bo.agent_id} belief on {ticker}",
                            "url": f"/api/v1/stores/beliefs/{bo.belief_id}",
                        })
                    lines.append("")
                else:
                    lines.append(f"No beliefs found for {ticker}.")
                    lines.append("")
            return ("\n".join(lines).strip(), citations)

        # No specific ticker — show summary
        total = belief_store.count()
        agents = sorted(belief_store.all_agents())
        all_tickers = sorted(belief_store.all_tickers())

        lines = [
            f"**Belief Store Summary**",
            f"- Total belief objects: {total}",
            f"- Active agents: {len(agents)} ({', '.join(agents[:6])}{'...' if len(agents) > 6 else ''})",
            f"- Tickers covered: {len(all_tickers)} ({', '.join(all_tickers[:8])}{'...' if len(all_tickers) > 8 else ''})",
        ]

        # Show latest beliefs
        latest = belief_store.query(limit=3)
        if latest:
            lines.append("")
            lines.append("**Most Recent Beliefs:**")
            for bo in latest:
                ts = bo.timestamp.strftime("%Y-%m-%d %H:%M")
                tks = set()
                for b in bo.beliefs:
                    tks.add(b.ticker)
                lines.append(f"- {bo.agent_id} ({ts}): {', '.join(sorted(tks))}")
                citations.append({
                    "type": "belief",
                    "id": str(bo.belief_id),
                    "label": f"{bo.agent_id} belief",
                    "url": f"/api/v1/stores/beliefs/{bo.belief_id}",
                })

        return ("\n".join(lines), citations)

    def _handle_regime(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle regime/risk mode queries."""
        run_store = self._state.run_store
        citations: list[dict[str, Any]] = []

        latest_run = run_store.get_latest(loop_type="main")
        if latest_run is None:
            return ("No pipeline runs found. Regime data is unavailable.", [])

        # Try to extract regime info from run metadata
        metadata = getattr(latest_run, "metadata", {}) or {}
        regime_state = metadata.get("regime_state", {})

        if isinstance(regime_state, dict):
            stat_regime = regime_state.get("statistical_regime", "Unknown")
            risk_mode = regime_state.get("system_risk_mode", "Unknown")
            confidence = regime_state.get("regime_confidence", 0.0)
            narrative = regime_state.get("narrative_label", "N/A")
        else:
            stat_regime = "Unknown"
            risk_mode = "NORMAL"
            confidence = 0.0
            narrative = "N/A"

        ts = latest_run.started_at.strftime("%Y-%m-%d %H:%M")
        lines = [
            f"**Current Market Regime** (as of {ts})",
            f"- Statistical regime: {stat_regime}",
            f"- Confidence: {confidence:.0%}" if isinstance(confidence, float) else f"- Confidence: {confidence}",
            f"- System risk mode: {risk_mode}",
            f"- Narrative: {narrative}",
        ]

        citations.append({
            "type": "pipeline",
            "id": str(latest_run.run_id),
            "label": f"Pipeline run {ts}",
            "url": f"/api/v1/pipeline/runs/{latest_run.run_id}",
        })

        return ("\n".join(lines), citations)

    def _handle_agents(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle agent health/status queries."""
        registry = self._state.agent_registry
        citations: list[dict[str, Any]] = []

        if not registry:
            return ("No agents are registered.", [])

        # Count by subsystem
        by_subsystem: dict[str, int] = {}
        by_classification: dict[str, int] = {}
        for agent_id, agent in registry.items():
            subsystem = getattr(agent, "subsystem", "unknown")
            classification = getattr(agent, "classification", "unknown")
            by_subsystem[subsystem] = by_subsystem.get(subsystem, 0) + 1
            by_classification[classification] = by_classification.get(classification, 0) + 1

        lines = [
            f"**Agent Registry** ({len(registry)} agents)",
            "",
            "**By Subsystem:**",
        ]
        for sub, count in sorted(by_subsystem.items()):
            lines.append(f"- {sub}: {count}")

        lines.append("")
        lines.append("**By Classification:**")
        for cls_name, count in sorted(by_classification.items()):
            lines.append(f"- {cls_name}: {count}")

        # Health summary if available
        if self._state.health_service:
            try:
                report = self._state.health_service.check()
                summary = report.summary()
                agents_summary = summary.get("agents", {})
                if agents_summary:
                    lines.append("")
                    lines.append("**Health Status:**")
                    for status, cnt in agents_summary.items():
                        lines.append(f"- {status}: {cnt}")
            except Exception:
                pass

        citations.append({
            "type": "agent",
            "id": "all",
            "label": "All agents",
            "url": "/api/v1/agents",
        })

        return ("\n".join(lines), citations)

    def _handle_pipeline(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle pipeline run/cycle queries."""
        run_store = self._state.run_store
        citations: list[dict[str, Any]] = []

        total = run_store.count()
        if total == 0:
            return ("No pipeline runs recorded yet.", [])

        rate = run_store.success_rate()
        latest = run_store.get_latest()

        lines = [
            f"**Pipeline Summary**",
            f"- Total runs: {total}",
            f"- Overall success rate: {rate:.0%}",
        ]

        # Breakdown by loop type
        for loop_type in ["main", "exit", "learning", "governance"]:
            count = run_store.count(loop_type=loop_type)
            if count > 0:
                loop_rate = run_store.success_rate(loop_type=loop_type)
                lines.append(f"- {loop_type}: {count} runs ({loop_rate:.0%} success)")

        if latest:
            ts = latest.started_at.strftime("%Y-%m-%d %H:%M")
            status = latest.status.value if hasattr(latest.status, "value") else str(latest.status)
            lines.extend([
                "",
                f"**Latest Run:**",
                f"- ID: {latest.run_id}",
                f"- Loop: {latest.loop_type}",
                f"- Status: {status}",
                f"- Started: {ts}",
            ])
            # Stage summary
            stage_results = getattr(latest, "stage_results", []) or []
            if stage_results:
                succeeded = sum(1 for s in stage_results if s.status.value == "SUCCEEDED" or s.status == "SUCCEEDED")
                failed = sum(1 for s in stage_results if s.status.value == "FAILED" or s.status == "FAILED")
                skipped = len(stage_results) - succeeded - failed
                lines.append(f"- Stages: {succeeded} succeeded, {failed} failed, {skipped} skipped")

            citations.append({
                "type": "pipeline",
                "id": str(latest.run_id),
                "label": f"Latest {latest.loop_type} run",
                "url": f"/api/v1/pipeline/runs/{latest.run_id}",
            })

        return ("\n".join(lines), citations)

    def _handle_shadow(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle shadow mode/signal queries."""
        store = self._state.shadow_signal_store
        citations: list[dict[str, Any]] = []

        if store is None:
            return ("Shadow signal tracking is not initialized.", [])

        total = store.count()
        if total == 0:
            return ("No shadow signals recorded yet.", [])

        signals = store.get_all()
        approved = sum(1 for s in signals if s.approved)
        rejected = total - approved
        summaries = store.get_summaries()

        # Accuracy stats
        has_1d = [s for s in signals if s.realized_return_1d is not None]
        has_5d = [s for s in signals if s.realized_return_5d is not None]

        lines = [
            f"**Shadow Mode Summary**",
            f"- Total signals: {total}",
            f"- Approved: {approved} | Rejected: {rejected}",
            f"- Pipeline runs: {len(summaries)}",
        ]

        if has_1d:
            correct_1d = sum(
                1 for s in has_1d
                if (s.realized_return_1d > 0 and s.direction == "LONG")
                or (s.realized_return_1d < 0 and s.direction == "SHORT")
            )
            acc_1d = correct_1d / len(has_1d) if has_1d else 0
            lines.append(f"- 1-day accuracy: {acc_1d:.0%} ({len(has_1d)} signals)")

        if has_5d:
            correct_5d = sum(
                1 for s in has_5d
                if (s.realized_return_5d > 0 and s.direction == "LONG")
                or (s.realized_return_5d < 0 and s.direction == "SHORT")
            )
            acc_5d = correct_5d / len(has_5d) if has_5d else 0
            lines.append(f"- 5-day accuracy: {acc_5d:.0%} ({len(has_5d)} signals)")

        # Recent signals
        recent = signals[:5]
        if recent:
            lines.extend(["", "**Recent Signals:**"])
            for sig in recent:
                ts = sig.timestamp.strftime("%Y-%m-%d %H:%M")
                status = "approved" if sig.approved else "rejected"
                lines.append(f"- {sig.ticker} {sig.direction} ({status}) @ {ts}")

        citations.append({
            "type": "pipeline",
            "id": "shadow",
            "label": "Shadow mode stats",
            "url": "/api/v1/shadow/stats",
        })

        return ("\n".join(lines), citations)

    def _handle_general(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        """Handle unclassified queries with a helpful guide."""
        return (
            "I can help you with the following:\n\n"
            "- **Portfolio**: Ask about holdings, equity, exposure, P&L, drawdown\n"
            "- **Positions**: Query specific stocks (e.g., 'How is AAPL doing?') or all positions\n"
            "- **Beliefs**: See what agents think about specific tickers or the market\n"
            "- **Regime**: Check the current market regime and risk mode\n"
            "- **Agents**: View agent health, status, and subsystem breakdown\n"
            "- **Pipeline**: Check recent run history and success rates\n"
            "- **Shadow Mode**: Review shadow trading signals and accuracy\n\n"
            "Try asking something like 'Show me the portfolio' or 'What do agents think about NVDA?'",
            [],
        )
