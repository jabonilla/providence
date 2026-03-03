#!/usr/bin/env python3
"""End-to-End Pipeline Test — Session 32.

Runs the full Providence pipeline with real API data and validates
every stage from Perception through Execution.

Usage:
    cd providence/
    python scripts/e2e_test.py              # Full test with all stages
    python scripts/e2e_test.py --skip-llm   # Skip adaptive (LLM) agents
    python scripts/e2e_test.py --ticker AAPL # Single ticker only

Requires:
    - .env file with API keys (Anthropic, Polygon, FRED, Alpaca)
    - Python 3.12+ with all dependencies installed
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("WARNING: python-dotenv not installed, using environment as-is")


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    keys = {
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "POLYGON_API_KEY": bool(os.getenv("POLYGON_API_KEY")),
        "FRED_API_KEY": bool(os.getenv("FRED_API_KEY")),
        "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
        "ALPACA_SECRET_KEY": bool(os.getenv("ALPACA_SECRET_KEY")),
    }
    return keys


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(label: str, status: str, detail: str = "") -> None:
    icon = "✅" if status == "OK" else "❌" if status == "FAIL" else "⚠️"
    line = f"  {icon} {label}: {status}"
    if detail:
        line += f" — {detail}"
    print(line)


async def test_perception(ticker: str) -> dict:
    """Test Phase 1: Perception — ingest real market data."""
    print_section("Phase 1: PERCEPTION")
    results = {}

    # Test PERCEPT-PRICE
    try:
        from providence.agents.perception.price import PerceptPrice
        from providence.infra.polygon_client import PolygonClient

        client = PolygonClient(api_key=os.getenv("POLYGON_API_KEY", ""))
        agent = PerceptPrice(polygon_client=client)

        from providence.agents.base import AgentContext
        ctx = AgentContext(
            agent_id="PERCEPT-PRICE",
            trigger="e2e-test",
            fragments=[],
            context_window_hash="e2e-test",
            timestamp=datetime.now(timezone.utc),
            metadata={"tickers": [ticker], "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
        )
        fragment = await agent.process(ctx)
        results["PERCEPT-PRICE"] = ("OK", f"Got {ticker} price data")
        print_result("PERCEPT-PRICE", "OK", f"Fragment for {ticker}")
    except Exception as e:
        results["PERCEPT-PRICE"] = ("FAIL", str(e)[:100])
        print_result("PERCEPT-PRICE", "FAIL", str(e)[:100])

    # Test PERCEPT-MACRO (FRED)
    try:
        from providence.agents.perception.macro import PerceptMacro
        from providence.infra.fred_client import FredClient

        fred = FredClient(api_key=os.getenv("FRED_API_KEY", ""))
        agent = PerceptMacro(fred_client=fred)
        ctx = AgentContext(
            agent_id="PERCEPT-MACRO",
            trigger="e2e-test",
            fragments=[],
            context_window_hash="e2e-test",
            timestamp=datetime.now(timezone.utc),
            metadata={"date": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
        )
        fragment = await agent.process(ctx)
        results["PERCEPT-MACRO"] = ("OK", "Got macro data from FRED")
        print_result("PERCEPT-MACRO", "OK", "Macro data from FRED")
    except Exception as e:
        results["PERCEPT-MACRO"] = ("FAIL", str(e)[:100])
        print_result("PERCEPT-MACRO", "FAIL", str(e)[:100])

    return results


async def test_full_pipeline(ticker: str, skip_llm: bool = False) -> dict:
    """Test Phase 2: Full pipeline via Orchestrator."""
    print_section("Phase 2: FULL PIPELINE (Orchestrator)")

    from providence.factory import build_agent_registry, build_agent_registry_from_env
    from providence.orchestration.orchestrator import Orchestrator
    from providence.orchestration.runner import ProvidenceRunner
    from providence.config.agent_config import AgentConfigRegistry
    from providence.services.context_svc import ContextService
    from providence.storage.fragment_store import FragmentStore
    from providence.storage.belief_store import BeliefStore
    from providence.storage.run_store import RunStore

    # Build infrastructure
    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    data_dir.mkdir(parents=True, exist_ok=True)

    fragment_store = FragmentStore(persist_path=data_dir / "fragments.jsonl")
    belief_store = BeliefStore(persist_path=data_dir / "beliefs.jsonl")
    run_store = RunStore(persist_path=data_dir / "runs.jsonl")

    config_registry = AgentConfigRegistry()
    context_svc = ContextService(config_registry)

    # Build agent registry (auto-discovers API keys from env)
    skip_perception = True  # We'll inject fragments manually
    registry = build_agent_registry_from_env(
        skip_perception=skip_perception,
        skip_adaptive=skip_llm,
    )

    agent_count = len(registry)
    print(f"  Agents in registry: {agent_count}")
    print(f"  Skip LLM agents: {skip_llm}")
    print(f"  Skip perception: {skip_perception}")

    orchestrator = Orchestrator(
        agent_registry=registry,
        context_service=context_svc,
        config_registry=config_registry,
        default_timeout=180.0,
    )

    runner = ProvidenceRunner(
        orchestrator=orchestrator,
        fragment_store=fragment_store,
        belief_store=belief_store,
        run_store=run_store,
    )

    # Run perception first to populate fragment store
    print("\n  Running perception sweep...")
    try:
        from providence.services.perception_scheduler import PerceptionScheduler
        from providence.config.watchlist import Watchlist

        percept_ids = {
            "PERCEPT-PRICE", "PERCEPT-FILING", "PERCEPT-NEWS",
            "PERCEPT-OPTIONS", "PERCEPT-CDS", "PERCEPT-MACRO",
        }
        perception_registry = build_agent_registry_from_env(
            skip_adaptive=True,
            agent_filter=percept_ids,
        )

        # Build a minimal watchlist with just our test ticker
        watchlist = Watchlist.from_dict({
            "max_positions": 5,
            "tickers": [{"ticker": ticker, "sector": "Technology", "priority": 1}],
        })

        scheduler = PerceptionScheduler(
            perception_agents=perception_registry,
            fragment_store=fragment_store,
            watchlist=watchlist,
        )

        stats = await scheduler.run_single(ticker)
        print(f"  Perception sweep: {stats.get('fragments', 0)} fragments, "
              f"{stats.get('errors', 0)} errors")
    except Exception as e:
        print(f"  ⚠️ Perception sweep failed: {e}")
        import traceback; traceback.print_exc()
        print("  Continuing with empty fragment store...")

    # Run main loop
    print("\n  Running main loop (Cognition → Regime → Decision → Execution)...")
    start = time.monotonic()

    try:
        runs = await runner.run_once(
            run_exit=True,
            run_governance=True,
        )
        elapsed = time.monotonic() - start
        print(f"\n  Pipeline completed in {elapsed:.1f}s\n")

        # Analyze results
        results = {}
        for loop_name, run in runs.items():
            print(f"  --- {loop_name} Loop ---")
            print(f"  Status: {run.status.value}")
            for sr in run.stage_results:
                status = "OK" if sr.status.value == "SUCCEEDED" else sr.status.value
                detail = ""
                if sr.error:
                    detail = sr.error[:80]
                elif sr.output:
                    detail = f"output keys: {list(sr.output.keys())[:5]}"
                print_result(sr.agent_id, status, detail)
                results[sr.agent_id] = (status, detail)
            print()

        # Summary stats
        total = sum(len(r.stage_results) for r in runs.values())
        succeeded = sum(
            1 for r in runs.values()
            for sr in r.stage_results
            if sr.status.value == "SUCCEEDED"
        )
        failed = total - succeeded
        print(f"  Total stages: {total}")
        print(f"  Succeeded: {succeeded}")
        print(f"  Failed/Skipped: {failed}")

        # Check stores
        belief_count = belief_store.count()
        fragment_count = fragment_store.count()
        run_count = run_store.count(loop_type=None)
        print(f"  Fragments stored: {fragment_count}")
        print(f"  Beliefs stored: {belief_count}")
        print(f"  Runs stored: {run_count}")

        return results

    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  ❌ Pipeline failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {"PIPELINE": ("FAIL", str(e)[:200])}


async def test_alpaca_paper() -> dict:
    """Test Phase 3: Alpaca paper trading connectivity."""
    print_section("Phase 3: ALPACA PAPER TRADING")

    try:
        from providence.infra.alpaca_client import AlpacaClient

        client = AlpacaClient(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            paper=True,
        )

        # Test account info
        account = await client.get_account()
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        print_result("Account", "OK", f"Equity: ${equity:,.2f}, Cash: ${cash:,.2f}")

        # Test market clock
        clock = await client.get_clock()
        is_open = clock.get("is_open", False)
        print_result("Market Clock", "OK", f"Market open: {is_open}")

        # Test positions
        positions = await client.list_positions()
        print_result("Positions", "OK", f"{len(positions)} open positions")

        # Test orders
        orders = await client.list_orders(status="open")
        print_result("Orders", "OK", f"{len(orders)} open orders")

        return {
            "ALPACA_ACCOUNT": ("OK", f"${equity:,.2f}"),
            "ALPACA_CLOCK": ("OK", str(is_open)),
            "ALPACA_POSITIONS": ("OK", str(len(positions))),
        }

    except Exception as e:
        print_result("Alpaca", "FAIL", str(e)[:100])
        return {"ALPACA": ("FAIL", str(e)[:100])}


async def main():
    parser = argparse.ArgumentParser(description="Providence E2E Pipeline Test")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to test (default: AAPL)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip adaptive/LLM agents")
    parser.add_argument("--skip-alpaca", action="store_true", help="Skip Alpaca test")
    parser.add_argument("--skip-perception", action="store_true", help="Skip perception test")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PROVIDENCE E2E PIPELINE TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ticker: {args.ticker}")
    print("=" * 60)

    # Check API keys
    print_section("API Key Check")
    keys = check_api_keys()
    for name, present in keys.items():
        print_result(name, "OK" if present else "MISSING")

    missing = [k for k, v in keys.items() if not v]
    if missing:
        print(f"\n  ⚠️ Missing keys: {', '.join(missing)}")
        print("  Some tests may fail.\n")

    all_results = {}

    # Phase 1: Perception
    if not args.skip_perception:
        perception_results = await test_perception(args.ticker)
        all_results.update(perception_results)

    # Phase 2: Full pipeline
    pipeline_results = await test_full_pipeline(args.ticker, skip_llm=args.skip_llm)
    all_results.update(pipeline_results)

    # Phase 3: Alpaca
    if not args.skip_alpaca:
        alpaca_results = await test_alpaca_paper()
        all_results.update(alpaca_results)

    # Final Summary
    print_section("FINAL SUMMARY")
    ok_count = sum(1 for s, _ in all_results.values() if s == "OK")
    fail_count = sum(1 for s, _ in all_results.values() if s in ("FAIL", "FAILED"))
    skip_count = sum(1 for s, _ in all_results.values() if s == "SKIPPED")
    total = len(all_results)

    print(f"  Total: {total}")
    print(f"  ✅ Passed: {ok_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"  ⚠️ Skipped: {skip_count}")
    print(f"\n  Success rate: {ok_count/max(total,1)*100:.0f}%\n")

    if fail_count > 0:
        print("  Failed stages:")
        for name, (status, detail) in all_results.items():
            if status in ("FAIL", "FAILED"):
                print(f"    ❌ {name}: {detail}")
        print()

    # Save results
    results_path = PROJECT_ROOT / "data" / "e2e_test" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": args.ticker,
            "skip_llm": args.skip_llm,
            "results": {k: {"status": s, "detail": d} for k, (s, d) in all_results.items()},
            "summary": {
                "total": total,
                "passed": ok_count,
                "failed": fail_count,
                "skipped": skip_count,
            },
        }, f, indent=2)
    print(f"  Results saved to: {results_path}\n")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
