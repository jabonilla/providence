#!/usr/bin/env python3
"""Shadow Mode Report Generator — evaluates signal quality from shadow runs.

Reads shadow signals from the signal store and generates a performance
report. This is the primary tool for Launch Plan Phase B success criteria:
  1. Directional accuracy > 55% at 5-day horizon
  2. Hypothetical Sharpe > 0.5
  3. No catastrophic drawdowns in simulated portfolio
  4. Stable signal generation (no empty runs)

Usage:
    python scripts/shadow_report.py [--data-dir data] [--format text|json]

Requires: providence package importable (run from project root with venv active).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from providence.schemas.enums import Direction, SystemMode
from providence.schemas.shadow import ShadowPerformanceReport, ShadowSignal
from providence.services.shadow_execution import ShadowSignalStore


def compute_report(store: ShadowSignalStore) -> dict[str, Any]:
    """Compute a shadow performance report from stored signals.

    Returns a dict suitable for ShadowPerformanceReport construction
    or direct JSON output.
    """
    signals = store.get_all()
    summaries = store.get_summaries()

    if not signals:
        return {
            "status": "NO_DATA",
            "message": "No shadow signals found. Run the pipeline in SHADOW mode first.",
            "total_runs": 0,
            "total_signals": 0,
        }

    # Basic counts
    total = len(signals)
    approved = [s for s in signals if s.approved]
    rejected = [s for s in signals if not s.approved]
    longs = [s for s in signals if s.direction == Direction.LONG]
    shorts = [s for s in signals if s.direction == Direction.SHORT]

    # Unique tickers and runs
    tickers = set(s.ticker for s in signals)
    run_ids = set(s.run_id for s in signals)

    # Confidence stats
    confidences = [s.confidence for s in signals if s.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Directional accuracy (only for signals with realized returns)
    accuracy_1d = _compute_accuracy(signals, "realized_return_1d")
    accuracy_5d = _compute_accuracy(signals, "realized_return_5d")
    accuracy_20d = _compute_accuracy(signals, "realized_return_20d")

    # Hypothetical returns
    hyp_1d = _compute_hypothetical_return(signals, "realized_return_1d")
    hyp_5d = _compute_hypothetical_return(signals, "realized_return_5d")
    hyp_20d = _compute_hypothetical_return(signals, "realized_return_20d")

    # Phase B success criteria
    meets_accuracy = accuracy_5d is not None and accuracy_5d > 0.55
    meets_sharpe = False  # Need more data for Sharpe calculation
    meets_stability = len(run_ids) >= 2  # At least 2 runs completed

    # Per-ticker breakdown
    ticker_stats = {}
    for ticker in sorted(tickers):
        ticker_signals = store.get_by_ticker(ticker)
        ticker_approved = [s for s in ticker_signals if s.approved]
        ticker_stats[ticker] = {
            "total": len(ticker_signals),
            "approved": len(ticker_approved),
            "avg_confidence": (
                sum(s.confidence for s in ticker_signals) / len(ticker_signals)
                if ticker_signals else 0.0
            ),
            "longs": sum(1 for s in ticker_signals if s.direction == Direction.LONG),
            "shorts": sum(1 for s in ticker_signals if s.direction == Direction.SHORT),
        }

    # Timestamps
    timestamps = [s.timestamp for s in signals]
    period_start = min(timestamps) if timestamps else None
    period_end = max(timestamps) if timestamps else None

    return {
        "status": "OK",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": period_start.isoformat() if period_start else None,
        "period_end": period_end.isoformat() if period_end else None,
        "total_runs": len(run_ids),
        "total_signals": total,
        "total_approved": len(approved),
        "total_rejected": len(rejected),
        "total_longs": len(longs),
        "total_shorts": len(shorts),
        "unique_tickers": len(tickers),
        "avg_confidence": round(avg_confidence, 4),
        "avg_signals_per_run": round(total / len(run_ids), 1) if run_ids else 0,
        "long_short_ratio": (
            round(len(longs) / len(shorts), 2) if shorts else None
        ),
        "accuracy_1d": round(accuracy_1d, 4) if accuracy_1d is not None else None,
        "accuracy_5d": round(accuracy_5d, 4) if accuracy_5d is not None else None,
        "accuracy_20d": round(accuracy_20d, 4) if accuracy_20d is not None else None,
        "hypothetical_return_1d": round(hyp_1d, 6) if hyp_1d is not None else None,
        "hypothetical_return_5d": round(hyp_5d, 6) if hyp_5d is not None else None,
        "hypothetical_return_20d": round(hyp_20d, 6) if hyp_20d is not None else None,
        "phase_b_criteria": {
            "accuracy_gt_55pct": meets_accuracy,
            "sharpe_gt_0_5": meets_sharpe,
            "stability_ok": meets_stability,
            "ready_for_paper": meets_accuracy and meets_sharpe and meets_stability,
        },
        "ticker_breakdown": ticker_stats,
    }


def _compute_accuracy(
    signals: list[ShadowSignal], return_field: str
) -> float | None:
    """Compute directional accuracy for signals with realized returns.

    A signal is "correct" if:
      - LONG and realized return > 0
      - SHORT and realized return < 0
    """
    evaluated = []
    for s in signals:
        if not s.approved:
            continue
        ret = getattr(s, return_field, None)
        if ret is None:
            continue
        if s.direction == Direction.LONG:
            evaluated.append(ret > 0)
        elif s.direction == Direction.SHORT:
            evaluated.append(ret < 0)

    if not evaluated:
        return None
    return sum(evaluated) / len(evaluated)


def _compute_hypothetical_return(
    signals: list[ShadowSignal], return_field: str
) -> float | None:
    """Compute average hypothetical return weighted by adjusted_weight."""
    weighted_returns = []
    total_weight = 0.0

    for s in signals:
        if not s.approved:
            continue
        ret = getattr(s, return_field, None)
        if ret is None:
            continue
        weight = s.adjusted_weight if s.adjusted_weight > 0 else 0.01
        # Short signals: profit from decline
        if s.direction == Direction.SHORT:
            ret = -ret
        weighted_returns.append(ret * weight)
        total_weight += weight

    if not weighted_returns or total_weight == 0:
        return None
    return sum(weighted_returns) / total_weight


def format_text_report(report: dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PROVIDENCE — Shadow Mode Performance Report")
    lines.append("=" * 70)

    if report.get("status") == "NO_DATA":
        lines.append(f"\n  {report['message']}")
        return "\n".join(lines)

    lines.append(f"\n  Generated: {report['generated_at']}")
    if report["period_start"]:
        lines.append(f"  Period:    {report['period_start']} → {report['period_end']}")

    lines.append(f"\n  --- Signal Summary ---")
    lines.append(f"  Total runs:       {report['total_runs']}")
    lines.append(f"  Total signals:    {report['total_signals']}")
    lines.append(f"  Approved:         {report['total_approved']}")
    lines.append(f"  Rejected:         {report['total_rejected']}")
    lines.append(f"  Longs:            {report['total_longs']}")
    lines.append(f"  Shorts:           {report['total_shorts']}")
    lines.append(f"  Unique tickers:   {report['unique_tickers']}")
    lines.append(f"  Avg confidence:   {report['avg_confidence']:.2%}")
    lines.append(f"  Avg signals/run:  {report['avg_signals_per_run']}")

    if report.get("long_short_ratio") is not None:
        lines.append(f"  Long/Short ratio: {report['long_short_ratio']:.2f}")

    lines.append(f"\n  --- Directional Accuracy ---")
    for horizon in ["1d", "5d", "20d"]:
        acc = report.get(f"accuracy_{horizon}")
        if acc is not None:
            lines.append(f"  {horizon:>3} accuracy:    {acc:.1%}")
        else:
            lines.append(f"  {horizon:>3} accuracy:    N/A (no realized returns yet)")

    lines.append(f"\n  --- Hypothetical Returns ---")
    for horizon in ["1d", "5d", "20d"]:
        ret = report.get(f"hypothetical_return_{horizon}")
        if ret is not None:
            lines.append(f"  {horizon:>3} return:      {ret:+.4%}")
        else:
            lines.append(f"  {horizon:>3} return:      N/A")

    lines.append(f"\n  --- Phase B Readiness ---")
    criteria = report["phase_b_criteria"]
    for key, met in criteria.items():
        icon = "✓" if met else "✗"
        lines.append(f"  [{icon}] {key}")

    if criteria["ready_for_paper"]:
        lines.append(f"\n  ★ READY FOR PAPER TRADING (Phase C) ★")
    else:
        lines.append(f"\n  ⚠ Not yet ready for paper trading")

    # Ticker breakdown
    ticker_stats = report.get("ticker_breakdown", {})
    if ticker_stats:
        lines.append(f"\n  --- Per-Ticker Breakdown ---")
        lines.append(f"  {'Ticker':<8} {'Total':>6} {'Appvd':>6} {'Conf':>7} {'L/S':>5}")
        lines.append(f"  {'-'*40}")
        for ticker, stats in ticker_stats.items():
            ls = f"{stats['longs']}/{stats['shorts']}"
            lines.append(
                f"  {ticker:<8} {stats['total']:>6} {stats['approved']:>6} "
                f"{stats['avg_confidence']:>6.1%} {ls:>5}"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Providence Shadow Mode Report Generator"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing shadow_signals.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    signal_path = args.data_dir / "shadow_signals.jsonl"
    store = ShadowSignalStore(persist_path=signal_path)

    report = compute_report(store)

    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_text_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
