#!/usr/bin/env python3
"""Pipeline health check script.

Runs the full Phase 1 pipeline for a single ticker and reports:
  - Schema validation pass/fail
  - Response parse success
  - Latency per step
  - Fragment counts
  - Belief quality metrics

Usage:
  python scripts/health_check.py              # defaults to AAPL with mock
  python scripts/health_check.py --ticker NVDA
  python scripts/health_check.py --live        # uses real Anthropic API (requires ANTHROPIC_API_KEY)

Spec Reference: Technical Spec v2.3, Session 7
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providence.agents.base import AgentContext
from providence.agents.cognition.fundamental import CognitFundamental
from providence.agents.cognition.response_parser import parse_llm_response
from providence.config.agent_config import AgentConfig, AgentConfigRegistry
from providence.infra.llm_client import AnthropicClient
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import DataType
from providence.services.context_svc import ContextService


# ---------------------------------------------------------------------------
# Sample data (mirrors tests/fixtures/sample_fragments.py)
# ---------------------------------------------------------------------------
def _load_sample_fragments(ticker: str):
    """Load sample fragments for the given ticker."""
    from tests.fixtures.sample_fragments import FRAGMENTS_BY_TICKER
    return FRAGMENTS_BY_TICKER.get(ticker, [])


def _load_mock_response(ticker: str):
    """Load mock LLM response for the given ticker."""
    from tests.fixtures.sample_beliefs import BELIEF_RESPONSES_BY_TICKER
    return BELIEF_RESPONSES_BY_TICKER.get(ticker)


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------
class MockLLMClient:
    """Mock LLM that returns canned responses for testing."""

    def __init__(self, response: dict):
        self._response = response

    async def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
        await asyncio.sleep(0.01)  # Simulate tiny latency
        return self._response


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
async def run_health_check(ticker: str, live: bool = False) -> dict:
    """Run the full pipeline and return a health report."""
    report = {
        "ticker": ticker,
        "live_mode": live,
        "steps": {},
        "overall": "PASS",
    }

    # Step 1: Load fragments
    t0 = time.perf_counter()
    fragments = _load_sample_fragments(ticker)
    dt = time.perf_counter() - t0

    report["steps"]["load_fragments"] = {
        "status": "PASS" if fragments else "FAIL",
        "count": len(fragments),
        "latency_ms": round(dt * 1000, 2),
    }

    if not fragments:
        report["overall"] = "FAIL"
        print(f"[FAIL] No sample fragments for ticker {ticker}")
        return report

    print(f"[PASS] Loaded {len(fragments)} fragments for {ticker} ({dt*1000:.1f}ms)")

    # Step 2: Assemble context
    t0 = time.perf_counter()
    try:
        registry = AgentConfigRegistry.from_dict({
            "COGNIT-FUNDAMENTAL": {
                "consumes": [
                    DataType.PRICE_OHLCV,
                    DataType.FILING_10K,
                    DataType.FILING_10Q,
                    DataType.FILING_8K,
                ],
                "max_token_budget": 100_000,
                "entity_scope": [],
                "peer_count": 0,
                "priority_window_hours": 72,
            }
        })
        svc = ContextService(registry)
        ctx = svc.assemble_context(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="health_check",
            available_fragments=fragments,
        )
        dt = time.perf_counter() - t0

        report["steps"]["context_assembly"] = {
            "status": "PASS",
            "fragment_count": len(ctx.fragments),
            "context_hash": ctx.context_window_hash[:16] + "...",
            "latency_ms": round(dt * 1000, 2),
        }
        print(f"[PASS] Context assembled: {len(ctx.fragments)} fragments, hash={ctx.context_window_hash[:16]}... ({dt*1000:.1f}ms)")

    except Exception as e:
        dt = time.perf_counter() - t0
        report["steps"]["context_assembly"] = {
            "status": "FAIL",
            "error": str(e),
            "latency_ms": round(dt * 1000, 2),
        }
        report["overall"] = "FAIL"
        print(f"[FAIL] Context assembly: {e}")
        return report

    # Step 3: Run COGNIT-FUNDAMENTAL
    t0 = time.perf_counter()
    try:
        if live:
            llm_client = AnthropicClient()
            agent = CognitFundamental(llm_client=llm_client, prompt_version="cognit_fundamental_v1.1.yaml")
        else:
            mock_response = _load_mock_response(ticker)
            if not mock_response:
                print(f"[SKIP] No mock response for {ticker}, skipping LLM step")
                report["steps"]["cognit_fundamental"] = {"status": "SKIP"}
                return report
            llm_client = MockLLMClient(mock_response)
            agent = CognitFundamental(llm_client=llm_client)

        belief_obj = await agent.process(ctx)
        dt = time.perf_counter() - t0

        report["steps"]["cognit_fundamental"] = {
            "status": "PASS",
            "belief_count": len(belief_obj.beliefs),
            "latency_ms": round(dt * 1000, 2),
        }
        print(f"[PASS] COGNIT-FUNDAMENTAL: {len(belief_obj.beliefs)} beliefs ({dt*1000:.1f}ms)")

    except Exception as e:
        dt = time.perf_counter() - t0
        report["steps"]["cognit_fundamental"] = {
            "status": "FAIL",
            "error": str(e),
            "latency_ms": round(dt * 1000, 2),
        }
        report["overall"] = "FAIL"
        print(f"[FAIL] COGNIT-FUNDAMENTAL: {e}")
        return report

    # Step 4: Validate beliefs
    issues = []

    for i, belief in enumerate(belief_obj.beliefs):
        # Check direction
        if belief.direction.value not in ("LONG", "SHORT", "NEUTRAL"):
            issues.append(f"Belief {i}: invalid direction {belief.direction}")

        # Check confidence range
        if belief.raw_confidence < 0 or belief.raw_confidence > 1:
            issues.append(f"Belief {i}: confidence out of range: {belief.raw_confidence}")

        # Check invalidation conditions
        if len(belief.invalidation_conditions) < 2:
            issues.append(f"Belief {i}: only {len(belief.invalidation_conditions)} invalidation conditions (need >= 2)")

        for cond in belief.invalidation_conditions:
            if not cond.metric:
                issues.append(f"Belief {i}: invalidation condition missing metric")

        # Check time horizon
        if belief.time_horizon_days < 30 or belief.time_horizon_days > 180:
            issues.append(f"Belief {i}: time horizon {belief.time_horizon_days} outside 30-180 range")

        # Check evidence
        context_frag_ids = {f.fragment_id for f in ctx.fragments}
        for ref in belief.evidence:
            if ref.source_fragment_id not in context_frag_ids:
                issues.append(f"Belief {i}: evidence ref {ref.source_fragment_id} not in context")

    report["steps"]["validation"] = {
        "status": "PASS" if not issues else "WARN",
        "issues": issues,
    }

    if issues:
        print(f"[WARN] Validation: {len(issues)} issues found")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"[PASS] Validation: all checks passed")

    # Step 5: Health status
    health = agent.get_health()
    report["steps"]["agent_health"] = {
        "status": health.status.value,
        "error_count_24h": health.error_count_24h,
    }
    print(f"[INFO] Agent health: {health.status.value}")

    # Summary
    total_latency = sum(
        s.get("latency_ms", 0) for s in report["steps"].values()
        if isinstance(s, dict)
    )
    report["total_latency_ms"] = round(total_latency, 2)

    print(f"\n{'='*50}")
    print(f"Health Check: {report['overall']}")
    print(f"Ticker: {ticker}")
    print(f"Mode: {'LIVE' if live else 'MOCK'}")
    print(f"Total latency: {total_latency:.1f}ms")
    print(f"{'='*50}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Providence Phase 1 Pipeline Health Check")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to check (default: AAPL)")
    parser.add_argument("--live", action="store_true", help="Use real Anthropic API (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    report = asyncio.run(run_health_check(args.ticker, args.live))

    if report["overall"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
