#!/usr/bin/env python3
"""Providence Pre-Flight Validation Script — Launch Plan Phase A.

Runs 10 validation checks required before entering Shadow Mode.
No external API calls — purely structural and import verification.

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --verbose
"""

import importlib
import inspect
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VERBOSE = "--verbose" in sys.argv


def _ok(label: str, detail: str = "") -> None:
    line = f"  \u2705 {label}"
    if detail:
        line += f" \u2014 {detail}"
    print(line)


def _fail(label: str, detail: str = "") -> None:
    line = f"  \u274c {label}"
    if detail:
        line += f" \u2014 {detail}"
    print(line)


def _warn(label: str, detail: str = "") -> None:
    line = f"  \u26a0\ufe0f  {label}"
    if detail:
        line += f" \u2014 {detail}"
    print(line)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =====================================================================
# Check 1: All 35 agents importable
# =====================================================================
def check_agent_imports() -> tuple[int, int]:
    _section("Check 1: Agent Imports (35 expected)")
    passed, failed = 0, 0

    agents = {
        # Perception (6)
        "providence.agents.perception.price": "PerceptPrice",
        "providence.agents.perception.filing": "PerceptFiling",
        "providence.agents.perception.news": "PerceptNews",
        "providence.agents.perception.options": "PerceptOptions",
        "providence.agents.perception.cds": "PerceptCds",
        "providence.agents.perception.macro": "PerceptMacro",
        # Cognition (6)
        "providence.agents.cognition.fundamental": "CognitFundamental",
        "providence.agents.cognition.technical": "CognitTechnical",
        "providence.agents.cognition.narrative": "CognitNarrative",
        "providence.agents.cognition.macro": "CognitMacro",
        "providence.agents.cognition.event": "CognitEvent",
        "providence.agents.cognition.crosssec": "CognitCrossSec",
        # Regime (4)
        "providence.agents.regime.stat": "RegimeStat",
        "providence.agents.regime.sector": "RegimeSector",
        "providence.agents.regime.narrative": "RegimeNarr",
        "providence.agents.regime.mismatch": "RegimeMismatch",
        # Decision (2)
        "providence.agents.decision.synth": "DecideSynth",
        "providence.agents.decision.optim": "DecideOptim",
        # Execution (4)
        "providence.agents.execution.validate": "ExecValidate",
        "providence.agents.execution.router": "ExecRouter",
        "providence.agents.execution.guardian": "ExecGuardian",
        "providence.agents.execution.capture": "ExecCapture",
        # Exit (5)
        "providence.agents.exit.cognit_exit": "CognitExit",
        "providence.agents.exit.invalid_mon": "InvalidMon",
        "providence.agents.exit.thesis_renew": "ThesisRenew",
        "providence.agents.exit.shadow_exit": "ShadowExit",
        "providence.agents.exit.renew_mon": "RenewMon",
        # Learning (4)
        "providence.agents.learning.attrib": "LearnAttrib",
        "providence.agents.learning.calib": "LearnCalib",
        "providence.agents.learning.retrain": "LearnRetrain",
        "providence.agents.learning.backtest": "LearnBacktest",
        # Governance (4)
        "providence.agents.governance.capital": "GovernCapital",
        "providence.agents.governance.maturity": "GovernMaturity",
        "providence.agents.governance.oversight": "GovernOversight",
        "providence.agents.governance.policy": "GovernPolicy",
    }

    for module_path, class_name in agents.items():
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _ok(f"{class_name}")
            passed += 1
        except Exception as e:
            _fail(f"{class_name}", str(e)[:80])
            if VERBOSE:
                traceback.print_exc()
            failed += 1

    print(f"\n  Result: {passed}/35 agents importable")
    return passed, failed


# =====================================================================
# Check 2: All schemas importable
# =====================================================================
def check_schema_imports() -> tuple[int, int]:
    _section("Check 2: Schema Imports")
    passed, failed = 0, 0

    try:
        import providence.schemas as s
        exported = s.__all__
        _ok(f"schemas.__init__", f"{len(exported)} exports")
        passed += 1

        # Verify each export resolves
        for name in exported:
            try:
                obj = getattr(s, name)
                passed += 1
            except AttributeError:
                _fail(f"Missing export: {name}")
                failed += 1
    except Exception as e:
        _fail("schemas package", str(e)[:80])
        failed += 1

    print(f"\n  Result: {passed} schema items verified")
    return passed, failed


# =====================================================================
# Check 3: Factory builds agent registry
# =====================================================================
def check_factory() -> tuple[int, int]:
    _section("Check 3: Agent Factory")
    passed, failed = 0, 0

    try:
        from providence.factory import build_agent_registry, ALL_AGENT_IDS

        _ok(f"ALL_AGENT_IDS", f"{len(ALL_AGENT_IDS)} agents defined")
        if len(ALL_AGENT_IDS) == 35:
            _ok("Agent count matches spec (35)")
            passed += 1
        else:
            _fail(f"Expected 35, got {len(ALL_AGENT_IDS)}")
            failed += 1

        # Build with everything skipped (no API keys needed)
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        frozen_count = len(registry)
        _ok(f"Frozen-only build", f"{frozen_count} agents")
        passed += 1

        # Check expected frozen count (21)
        if frozen_count == 21:
            _ok("Frozen count matches spec (21)")
            passed += 1
        else:
            _warn(f"Expected 21 frozen, got {frozen_count}")

    except Exception as e:
        _fail("Factory build", str(e)[:80])
        if VERBOSE:
            traceback.print_exc()
        failed += 1

    return passed, failed


# =====================================================================
# Check 4: Orchestrator configuration
# =====================================================================
def check_orchestrator() -> tuple[int, int]:
    _section("Check 4: Orchestrator Configuration")
    passed, failed = 0, 0

    try:
        from providence.orchestration.orchestrator import Orchestrator
        from providence.services.context_svc import ContextService
        from providence.config.agent_config import AgentConfigRegistry
        from providence.factory import build_agent_registry

        registry = build_agent_registry(skip_perception=True, skip_adaptive=True)
        config = AgentConfigRegistry()
        ctx_svc = ContextService(config_registry=config)

        orch = Orchestrator(
            agent_registry=registry,
            context_service=ctx_svc,
            config_registry=config,
        )

        # Check all 4 loops are callable
        for loop in ["run_main_loop", "run_exit_loop", "run_learning_loop", "run_governance_loop"]:
            if hasattr(orch, loop):
                _ok(f"Orchestrator.{loop}()")
                passed += 1
            else:
                _fail(f"Missing Orchestrator.{loop}()")
                failed += 1

    except Exception as e:
        _fail("Orchestrator init", str(e)[:80])
        if VERBOSE:
            traceback.print_exc()
        failed += 1

    return passed, failed


# =====================================================================
# Check 5: Storage layer
# =====================================================================
def check_storage() -> tuple[int, int]:
    _section("Check 5: Storage Layer")
    passed, failed = 0, 0

    for store_name in ["FragmentStore", "BeliefStore", "RunStore"]:
        try:
            mod = importlib.import_module(f"providence.storage.{store_name.lower().replace('store', '_store')}")
            cls = getattr(mod, store_name)
            instance = cls()  # In-memory only
            _ok(f"{store_name} instantiated")
            passed += 1
        except Exception as e:
            _fail(f"{store_name}", str(e)[:80])
            failed += 1

    return passed, failed


# =====================================================================
# Check 6: API layer
# =====================================================================
def check_api() -> tuple[int, int]:
    _section("Check 6: REST API Layer")
    passed, failed = 0, 0

    try:
        from providence.api.app import create_app
        _ok("create_app importable")
        passed += 1
    except ImportError as e:
        if "fastapi" in str(e).lower():
            _warn("FastAPI not installed (optional dependency)")
            return passed, failed
        _fail("API import", str(e)[:80])
        failed += 1
        return passed, failed

    try:
        from providence.api.security import (
            require_api_key, RateLimitMiddleware,
            RequestSizeLimitMiddleware, sanitize_error
        )
        _ok("Security module importable")
        passed += 1
    except Exception as e:
        _fail("Security module", str(e)[:80])
        failed += 1

    # Check route modules
    for route_mod in ["health", "pipeline", "agents", "stores", "config"]:
        try:
            importlib.import_module(f"providence.api.routes.{route_mod}")
            _ok(f"Route: {route_mod}")
            passed += 1
        except Exception as e:
            _fail(f"Route: {route_mod}", str(e)[:80])
            failed += 1

    return passed, failed


# =====================================================================
# Check 7: Configuration files
# =====================================================================
def check_config() -> tuple[int, int]:
    _section("Check 7: Configuration Files")
    passed, failed = 0, 0

    config_files = {
        "agents.yaml": PROJECT_ROOT / "providence" / "config" / "agents.yaml",
        "watchlist.yaml": PROJECT_ROOT / "providence" / "config" / "watchlist.yaml",
        "Dockerfile": PROJECT_ROOT / "Dockerfile",
        "docker-compose.yml": PROJECT_ROOT / "docker-compose.yml",
        ".env.example": PROJECT_ROOT / ".env.example",
        "pyproject.toml": PROJECT_ROOT / "pyproject.toml",
    }

    for name, path in config_files.items():
        if path.exists():
            _ok(f"{name}", f"{path.stat().st_size} bytes")
            passed += 1
        else:
            _fail(f"{name} missing", str(path))
            failed += 1

    return passed, failed


# =====================================================================
# Check 8: Prompt templates
# =====================================================================
def check_prompts() -> tuple[int, int]:
    _section("Check 8: Prompt Templates")
    passed, failed = 0, 0

    prompts_dir = PROJECT_ROOT / "providence" / "prompts"
    if not prompts_dir.exists():
        _fail("prompts/ directory missing")
        return 0, 1

    templates = list(prompts_dir.glob("*.py"))
    _ok(f"Prompt template files", f"{len(templates)} found")
    passed += 1

    for t in templates:
        if t.stat().st_size > 0:
            passed += 1
        else:
            _fail(f"Empty template: {t.name}")
            failed += 1

    return passed, failed


# =====================================================================
# Check 9: API keys (environment)
# =====================================================================
def check_api_keys() -> tuple[int, int]:
    _section("Check 9: API Keys (Environment)")
    passed, failed = 0, 0

    keys = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude) — required for adaptive agents",
        "POLYGON_API_KEY": "Polygon.io — required for price/options/CDS/news",
        "FRED_API_KEY": "FRED — required for macro data",
        "ALPACA_API_KEY": "Alpaca Markets — required for trading",
        "ALPACA_SECRET_KEY": "Alpaca secret — required for trading",
    }

    for key, desc in keys.items():
        val = os.getenv(key, "")
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            _ok(f"{key}", f"set ({masked})")
            passed += 1
        else:
            _warn(f"{key} NOT SET", desc)

    return passed, failed


# =====================================================================
# Check 10: Codebase metrics
# =====================================================================
def check_codebase_metrics() -> tuple[int, int]:
    _section("Check 10: Codebase Metrics")
    passed, failed = 0, 0

    # Count Python files
    py_files = list(PROJECT_ROOT.rglob("*.py"))
    py_files = [f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)]
    src_files = [f for f in py_files if not str(f).startswith(str(PROJECT_ROOT / "tests"))]
    test_files = [f for f in py_files if str(f).startswith(str(PROJECT_ROOT / "tests"))]

    _ok(f"Source files", f"{len(src_files)}")
    _ok(f"Test files", f"{len(test_files)}")

    # Count lines
    total_lines = 0
    for f in py_files:
        try:
            total_lines += sum(1 for _ in open(f))
        except Exception:
            pass
    _ok(f"Total Python LOC", f"{total_lines:,}")
    passed += 1

    # Check docs
    docs_dir = PROJECT_ROOT / "docs"
    if docs_dir.exists():
        docs = list(docs_dir.glob("*.docx"))
        _ok(f"Documentation", f"{len(docs)} .docx files")
        for d in docs:
            _ok(f"  {d.name}", f"{d.stat().st_size:,} bytes")
        passed += 1

    return passed, failed


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    print("\n" + "=" * 60)
    print("  PROVIDENCE PRE-FLIGHT VALIDATION")
    print(f"  Launch Plan Phase A — 10 Checks")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    checks = [
        check_agent_imports,
        check_schema_imports,
        check_factory,
        check_orchestrator,
        check_storage,
        check_api,
        check_config,
        check_prompts,
        check_api_keys,
        check_codebase_metrics,
    ]

    for check_fn in checks:
        try:
            p, f = check_fn()
            total_passed += p
            total_failed += f
        except Exception as e:
            print(f"\n  \u274c Check crashed: {e}")
            if VERBOSE:
                traceback.print_exc()
            total_failed += 1

    # Summary
    _section("SUMMARY")
    print(f"  Total checks passed: {total_passed}")
    print(f"  Total checks failed: {total_failed}")
    print()

    if total_failed == 0:
        print("  \u2705 ALL CHECKS PASSED — Ready for Shadow Mode")
    else:
        print(f"  \u26a0\ufe0f  {total_failed} issues found — review before proceeding")

    print()
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
