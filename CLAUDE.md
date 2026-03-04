# Providence — AI-Native Hedge Fund

## Current Status
<!-- Update this after each session -->
Phase: 13 — Security Audit & Hardening (API auth, rate limiting, error sanitization, CORS lockdown)
Last completed session: 35 (Security Audit — API key auth, rate limiting, request size limits, error sanitization, CORS hardening, input validation, .dockerignore)
Implemented components: MarketStateFragment, BeliefObject, all enums, BaseAgent ABC, AgentContext, HealthStatus, exception hierarchy, hashing utils, structured logging, PricePayload, PolygonClient, PERCEPT-PRICE, FilingPayload, EdgarClient, filing parser, PERCEPT-FILING, AgentConfig, AgentConfigRegistry, token estimation, ContextService (7-step assembly), agents.yaml config, AnthropicClient, LLMClient Protocol, response_parser, CognitFundamental agent, prompt templates v1.0 + v1.1, sample fixtures (5 stocks), integration tests, health check script, API key redaction utility, NewsPayload, OptionsPayload, CdsPayload, MacroYieldPayload, MacroEconomicPayload, FredClient, PERCEPT-NEWS, PERCEPT-OPTIONS, PERCEPT-CDS, PERCEPT-MACRO, TechnicalSignals dataclass, technical indicator functions (SMA/EMA/RSI/MACD/Bollinger/momentum), compute_all_signals aggregator, CognitTechnical agent (FROZEN — zero LLM), signal-to-belief generation, technical invalidation conditions, CognitMacro agent (ADAPTIVE — Claude Sonnet 4), cognit_macro_v1.0 prompt template, macro test fixtures, CognitEvent agent (ADAPTIVE — Claude Sonnet 4), cognit_event_v1.0 prompt template, event test fixtures, CognitNarrative agent (ADAPTIVE — Claude Sonnet 4, peer comparison), cognit_narrative_v1.0 prompt template, narrative test fixtures, CognitCrossSec agent (ADAPTIVE — Claude Sonnet 4, peer comparison), cognit_crosssec_v1.0 prompt template, crosssec test fixtures, cognition __init__.py exports (6 agents), StatisticalRegime enum (4 states), SystemRiskMode enum (4 modes), RegimeStateObject schema, SectorRegimeOverlay schema, NarrativeRegimeOverlay schema, RegimeFeatures dataclass, regime feature extraction (realized vol, drawdown, macro), HMM model (pure Python, 4-state, forward algorithm), RegimeStat agent (FROZEN — zero LLM, HMM regime classification), sector_features module (ticker→GICS mapping, per-sector feature extraction, relative stress), RegimeSector agent (FROZEN — zero LLM, per-sector HMM classification), RegimeNarr agent (ADAPTIVE — Claude Sonnet 4, narrative regime classification), regime_narr_v1.0 prompt template, narrative regime test fixtures, parse_narrative_response parser, RegimeMismatch agent (FROZEN — zero LLM, stat-narrative-sector divergence detection, risk mode escalation), regime __init__.py exports (4 agents), Decision schemas (ContributingThesis, ConflictResolution, ActiveInvalidation, SynthesizedPositionIntent, SynthesisOutput, PortfolioMetadata, ProposedPosition, PositionProposal), Action enum, DecideSynth agent (ADAPTIVE — Claude Sonnet 4, belief synthesis with conflict resolution), decide_synth_v1.0 prompt template, parse_synthesis_response parser, synthesis test fixtures, DecideOptim agent (FROZEN — zero LLM, Black-Litterman portfolio optimization with regime-aware constraints), decision __init__.py exports (2 agents), Execution schemas (ValidationResult, ValidatedProposal, RoutedOrder, RoutingPlan, GuardianCheck, GuardianVerdict, TrailingStopState, CaptureDecision, CaptureOutput), ExecValidate agent (FROZEN — pre-trade validation with risk-mode constraints), ExecRouter agent (FROZEN — order routing with strategy/urgency/slippage), ExecGuardian agent (FROZEN — kill switch and circuit breakers), ExecCapture agent (FROZEN — trailing stop management with supremacy, 3-stage trim, hard giveback), execution __init__.py exports (4 agents), Exit schemas (ExitAssessment, ExitOutput, MonitoredCondition, InvalidationMonitorOutput, RenewalCandidate, ThesisRenewalOutput, ShadowExitSignal, ShadowExitOutput, BeliefHealthReport, RenewalMonitorOutput), CognitExit agent (ADAPTIVE — Claude Sonnet 4, exit assessment with renewal deferral), parse_exit_response parser, exit test fixtures, InvalidMon agent (FROZEN — invalidation condition monitoring with breach magnitude/velocity/confidence impact), ThesisRenew agent (FROZEN — thesis renewal with confidence decay, asymmetry scoring, regime alignment), ShadowExit agent (FROZEN — shadow exit tracking, COGNIT-EXIT vs EXEC-CAPTURE divergence detection), RenewMon agent (FROZEN — renewal monitoring with belief health scoring, urgency classification), exit __init__.py exports (5 agents), Learning schemas (AgentAttribution, TickerAttribution, AttributionOutput, CalibrationBucket, AgentCalibration, CalibrationOutput, RetrainRecommendation, RetrainOutput, BacktestPeriod, BacktestOutput), LearnAttrib agent (FROZEN — offline performance attribution with hit rate, IR, Sharpe contribution), LearnCalib agent (FROZEN — offline confidence calibration with Brier score, bucket analysis), LearnRetrain agent (FROZEN — offline retraining recommendations with priority levels, shadow mode enforcement), LearnBacktest agent (FROZEN — offline backtesting with sub-period analysis, annualized metrics, profit factor), learning __init__.py exports (4 agents), Governance enums (CapitalTier, MaturityStage, IncidentSeverity), Governance schemas (TierConstraints, CapitalTierOutput, AgentMaturityRecord, MaturityGateOutput, GovernanceIncident, SystemHealthSummary, OversightOutput, PolicyViolation, PolicyOutput), GovernCapital agent (FROZEN — AUM tier classification with execution constraints), GovernMaturity agent (FROZEN — agent deployment stage evaluation with promotion criteria), GovernOversight agent (FROZEN — system health aggregation with incident detection), GovernPolicy agent (FROZEN — policy enforcement with violation detection), governance __init__.py exports (4 agents), Orchestration models (StageResult, PipelineRun, StageStatus, RunStatus), PipelineStage (isolated async executor with timeout/error isolation), Orchestrator (DAG coordinator with 4 loop methods: main/exit/learning/governance), ProvidenceRunner (scheduler with run_once/run_continuous/run_learning_batch), OrchestrationError exception, Agent factory (build_agent_registry with dependency injection, skip_perception/skip_adaptive/agent_filter), CLI entry point (__main__.py with run-once/run-continuous/run-learning/health/list-agents commands), FragmentStore (append-only, in-memory + JSONL persistence, indexed by data_type/entity), BeliefStore (append-only, indexed by agent_id/ticker), RunStore (pipeline run history with success_rate), Storage-wired Runner (auto-pulls fragments from FragmentStore, extracts+stores BeliefObjects, persists all PipelineRuns to RunStore), CLI --data-dir for persistent storage, REST API (FastAPI app factory, AppState DI, 5 route modules, 17 response/request schemas, server entry point, CORS middleware, request logging), Deployment (multi-stage Dockerfile, docker-compose with nginx, GitHub Actions CI/CD, monitoring script), Security (API key auth middleware, rate limiting, request size limits, error sanitization, CORS hardening, .dockerignore)

## Quick Reference
- Spec: Technical Spec v2.3 (available in Claude Project "Providence")
- Language: Python 3.12+
- Schemas: Pydantic v2
- Testing: pytest, 80%+ coverage
- All schemas defined in providence/schemas/
- All agents in providence/agents/{subsystem}/
- Tests mirror source structure in tests/

## Architecture Invariants
- Frozen components = ZERO LLM calls
- Research agents never see each other's beliefs
- EXEC-CAPTURE trailing stop overrides everything
- Max 3 trim stages per position
- All data immutable after creation
- All invalidation conditions machine-evaluable
- No live learning — all retraining offline

## System Overview

Providence is an autonomous hedge fund system that uses multiple AI agents to identify, execute, manage, and exit equity positions. The core innovation is **structured disagreement**: investment decisions emerge from independent specialized agents whose views are synthesized through empirically-validated conflict resolution.

### 7 Subsystems
1. **Perception** (Frozen): Ingests market data from Polygon.io, SEC EDGAR, news, options, CDS, macro
2. **Cognition** (Adaptive): 7 independent Research Agents producing investment theses
3. **Regime** (Hybrid): HMM + LLM regime classification with sector overlays
4. **Decision** (Hybrid): Belief synthesis + Black-Litterman portfolio optimization
5. **Execution** (Frozen): Trade validation, routing, kill switch, trailing stop
6. **Learning** (Offline): Attribution, calibration, retraining — all offline only
7. **Governance** (Human): Capital tiers, agent maturity gates, oversight dashboard

## Core Schemas

### MarketStateFragment (Section 2.1)
- fragment_id (UUID), agent_id, timestamp, source_timestamp, version (SHA-256)
- entity, data_type (Enum from Data Type Registry), schema_version (SemVer)
- source_hash, validation_status (VALID/QUARANTINED/PARTIAL), payload (JSON)
- Content hash computed automatically via SHA-256 of serialized payload
- Immutable (frozen=True)

### BeliefObject (Section 2.2)
- belief_id, agent_id, timestamp, context_window_hash, beliefs array
- Each Belief: thesis_id, ticker, thesis_summary, direction (LONG/SHORT/NEUTRAL), magnitude (SMALL/MODERATE/LARGE), raw_confidence (0.0-1.0), time_horizon_days, evidence array, invalidation_conditions array
- InvalidationCondition: must be machine-evaluable with metric, operator (GT/LT/EQ/CROSSES_ABOVE/CROSSES_BELOW), threshold, breach_magnitude, breach_velocity

### RegimeStateObject (Section 2.3)
- statistical_regime: 4-state HMM (LOW_VOL_TRENDING, HIGH_VOL_MEAN_REVERTING, CRISIS_DISLOCATION, TRANSITION_UNCERTAIN)
- narrative_regime: label, confidence, key_signals, affected_sectors
- sector_overlays: Map of sector -> SectorRegimeOverlay
- system_risk_mode: NORMAL/CAUTIOUS/DEFENSIVE/HALTED

### SynthesizedPositionIntent (Section 2.4)
- ticker, net_direction, synthesized_confidence, contributing_theses, conflicting_theses
- conflict_resolution, time_horizon_days, regime_adjustment, active_invalidations

### PositionProposal (Section 2.5)
- proposals array of ProposedPosition with action (OPEN_LONG/OPEN_SHORT/CLOSE/ADJUST)
- portfolio_metadata: gross_exposure, net_exposure, sector_concentrations, estimated_sharpe

### ExitAssessment (Section 2.8)
- exit_confidence [0-1], regret_estimate_bps, regret_direction (MISSED_UPSIDE/SUFFERED_GIVEBACK)
- Multi-stage trim: max 3 stages, trim_pct applies to REMAINING position
- Stage 0: original logic; Stage 1: prior trim outcome influences; Stage 2+: default CLOSE

### TrailingStopState (Section 2.9)
- Activation: unrealized PnL > 2.0x expected_return
- Trail: 30% NORMAL, 20% CAUTIOUS/DEFENSIVE
- Hard giveback: > 50% of peak -> close
- Minimum hold: 5 days

## Agent Classification

### Frozen (ZERO LLM calls — pure computation):
- PERCEPT-PRICE, PERCEPT-FILING, PERCEPT-NEWS, PERCEPT-OPTIONS, PERCEPT-CDS, PERCEPT-MACRO
- COGNIT-TECHNICAL
- REGIME-STAT, REGIME-SECTOR, REGIME-MISMATCH
- DECIDE-OPTIM
- EXEC-VALIDATE, EXEC-ROUTER, EXEC-GUARDIAN, EXEC-CAPTURE
- INVALID-MON, THESIS-RENEW, SHADOW-EXIT, RENEW-MON
- LEARN-ATTRIB, LEARN-CALIB, LEARN-RETRAIN, LEARN-BACKTEST
- GOVERN-CAPITAL, GOVERN-MATURITY, GOVERN-OVERSIGHT, GOVERN-POLICY

### Adaptive (uses LLM — subject to offline retraining):
- COGNIT-FUNDAMENTAL (Claude Sonnet 4)
- COGNIT-NARRATIVE (GPT-4o)
- COGNIT-MACRO (Claude Sonnet 4)
- COGNIT-EVENT (GPT-4o)
- COGNIT-CROSSSEC (Claude Sonnet 4)
- COGNIT-EXIT (Claude Sonnet 4)
- REGIME-NARR (Claude Sonnet 4)
- DECIDE-SYNTH (Claude Sonnet 4)

## Implementation Phases

### Phase 1: Single Agent Pipeline (Sessions 1-7)
PERCEPT-PRICE -> MarketStateFragment -> CONTEXT-SVC -> COGNIT-FUNDAMENTAL -> BeliefObject

### Phase 2: Multi-Agent + Synthesis (Sessions 8-16)
All research agents, regime system, decision system, execution system

### Phase 3: Exit System (Sessions 17-21)
COGNIT-EXIT + INVALID-MON + THESIS-RENEW + SHADOW-EXIT + RENEW-MON — COMPLETE

### Phase 4: Learning + Governance (Sessions 22-23)
LEARN-ATTRIB + LEARN-CALIB + LEARN-RETRAIN + LEARN-BACKTEST — COMPLETE
GOVERN-CAPITAL + GOVERN-MATURITY + GOVERN-OVERSIGHT + GOVERN-POLICY — COMPLETE
ALL 7 SUBSYSTEMS BUILT. 35 agents total.

### Phase 5: Orchestration (Session 24)
StageResult + PipelineRun models, PipelineStage executor, Orchestrator (4 loops), ProvidenceRunner — COMPLETE
Pipeline: Cognition (6 parallel) → Regime (3 parallel + MISMATCH) → Decision (sequential) → Execution (strictly sequential)
Loops: Main (trading), Exit (5 sequential), Learning (4 sequential), Governance (4 sequential)
Error isolation: failed stages → FAILED result, downstream dependents → SKIPPED, independent stages continue

### Phase 5 continued: Factory + CLI (Session 25)
Agent factory (build_agent_registry) — dependency injection for 3 agent categories:
  Frozen (21 agents, zero args), Adaptive (8 agents, optional LLM client), Perception (6 agents, require API clients)
CLI entry point (python -m providence) — 5 commands: run-once, run-continuous, run-learning, health, list-agents
Supports: --skip-perception, --skip-adaptive, --timeout, --log-level, agent_filter

### Phase 6: Storage Layer (Session 26)
FragmentStore — append-only MarketStateFragment storage, in-memory with JSONL persistence
  Indexed by fragment_id, data_type, entity. Query: data_types, entities, validation_status, timestamp range
BeliefStore — append-only BeliefObject storage, indexed by belief_id, agent_id, ticker
  get_latest_by_agent(), get_latest_by_ticker() convenience methods
RunStore — PipelineRun history, indexed by run_id, loop_type. success_rate() analytics
All stores: thread-safe (RLock), deduplication by primary ID, newest-first results

### Phase 6 continued: Storage Wiring (Session 27)
ProvidenceRunner now accepts optional FragmentStore, BeliefStore, RunStore
  - fragments=None → auto-pulls from FragmentStore
  - Cognition outputs extracted and stored in BeliefStore
  - All PipelineRuns persisted to RunStore
  - CLI --data-dir flag for persistent storage directory (data/fragments.jsonl, beliefs.jsonl, runs.jsonl)

### Phase 7: Integration Testing (Session 28)
Full pipeline integration tests (tests/integration/test_full_pipeline.py):
  - Complete cycle: Main → Exit → Governance with all stores
  - Storage roundtrip: persist to disk, reload, run again
  - Learning batch with metadata flow
  - Continuous mode start/stop with fragment provider
  - Multi-cycle analytics (success_rate)
  - Agent count verification (35 total, 29 orchestrated)
Fixed: orchestrator fallback context hash (was passing strings to compute_context_window_hash expecting objects)

### Phase 7 continued: Polish (Session 28)
HealthService (system health aggregation: HEALTHY/DEGRADED/CRITICAL/HALTED), CLI health command upgraded
Package __init__.py with version, pyproject.toml CLI entry point

### Phase 8: Deployment + Bug Fixes (Session 29)
Fixed: Black-Litterman view signal multiplier 0.10→0.20 (short views now correctly negative)
Dockerfile (Python 3.12-slim, health check, data volume)
docker-compose.yml (continuous, manual, learning profiles)
Makefile (install, test, lint, run, docker commands)
.env.example (API key placeholders)
ALL TESTS GREEN (1392 passed, 0 failed)

### Phase 9: Broker Integration + Portfolio Management (Session 30)
AlpacaClient (providence/infra/alpaca_client.py) — async broker client for Alpaca Markets:
  Paper/live mode via ALPACA_PAPER env var (default: paper), submit_order, get_order, list_orders, cancel_order,
  list_positions, get_position, close_position, close_all_positions, get_account, get_portfolio_history, get_clock
  Retry logic with exponential backoff, rate limiting, structured logging
Watchlist (providence/config/watchlist.py) — ticker universe configuration:
  WatchlistEntry (ticker, sector, priority, tags), Watchlist (from_yaml, from_dict, default), config/watchlist.yaml (20 tickers)
  Queries: tickers, by_sector(), by_priority(), enabled_entries, max_positions
PortfolioTracker (providence/portfolio/tracker.py) — live position and P&L tracking:
  Position (qty, entry/current price, unrealized/realized P&L, weight, sector, days_held)
  PortfolioSnapshot (equity, cash, positions, gross/net/sector exposure, drawdown from peak)
  sync_from_broker(), record_fill(), update_price(), snapshot(), JSONL persistence
OrderManager (providence/portfolio/order_manager.py) — order lifecycle state machine:
  OrderStatus: PENDING→SUBMITTED→PARTIALLY_FILLED→FILLED (+ CANCELLED/REJECTED/EXPIRED/FAILED)
  ManagedOrder with immutable transition log, retry tracking, idempotent client_order_id
  create_from_routed_order() bridges EXEC-ROUTER output to broker (Action→Side, strategy→type, weight→notional)
  update_from_broker() maps Alpaca statuses to OrderStatus, JSONL persistence
ExecutionService (providence/services/execution_service.py) — pipeline-to-broker bridge:
  execute_routing_plan() consumes RoutingPlan + GuardianVerdict, submits to Alpaca, polls fills
  emergency_halt() cancels all orders on system_halt, reconcile() syncs with broker state
  retry_failed_orders() retries orders under max_retries limit
PerceptionScheduler (providence/services/perception_scheduler.py) — watchlist-driven data ingestion:
  run_full_sweep() processes all tickers, run_priority_sweep() high-priority only, run_single() one ticker
  Rate limiting between tickers/agents, failure isolation, sweep statistics

### Phase 10: E2E Pipeline Fix (Session 32)
Orchestrator metadata key fixes — 4 broken execution chain bridges:
  DECIDE-OPTIM output → EXEC-VALIDATE: was "optimization_output", now also aliased to "proposal"
  EXEC-VALIDATE output → EXEC-ROUTER: was "exec_exec-validate", now "validated_proposal"
  EXEC-ROUTER output → EXEC-GUARDIAN: was "exec_exec-router", now "routing_plan"
  EXEC-GUARDIAN output → EXEC-CAPTURE: was "exec_exec-guardian", now "guardian_verdict"
Regime state bridge: dict-first handling (PipelineStage serializes to dict), NORMAL fallback if missing
Synth bridge: dict-first extraction of position_intents from serialized SynthesisOutput
Exit/Learning/Governance loops: _LOOP_SEMANTIC_KEYS mapping for downstream agent metadata expectations
  COGNIT-EXIT → "exit_assessments", INVALID-MON → "invalidation_results", THESIS-RENEW → "renewal_state"
  LEARN-ATTRIB → "attribution_results", LEARN-CALIB → "calibration_results", LEARN-RETRAIN → "retrain_recommendations"
E2E test script (scripts/e2e_test.py) — 3-phase test: perception, full pipeline, Alpaca paper trading

### Phase 11: REST API Layer (Session 33)
FastAPI application factory (providence/api/app.py) — create_app() with CORS, request logging middleware, exception handlers
AppState dependency injection (providence/api/deps.py) — singleton container for stores, registry, services, runner
API response/request schemas (providence/api/schemas.py) — 17 Pydantic models decoupled from internal domain models
  SystemHealthResponse, AgentHealthResponse, PipelineRunResponse, StageResultResponse, RunTriggerRequest/Response
  AgentInfoResponse, FragmentSummaryResponse, FragmentDetailResponse, FragmentStoreStatsResponse
  BeliefSummaryResponse, BeliefDetailResponse, BeliefStoreStatsResponse, RunStoreStatsResponse
  WatchlistResponse, WatchlistEntryResponse, PositionResponse, PortfolioSnapshotResponse, ErrorResponse, StatusResponse
5 route modules under providence/api/routes/:
  health.py — GET /health (system summary), /health/ready (readiness probe), /health/live (liveness probe)
  pipeline.py — POST /pipeline/run (trigger cycle), POST /pipeline/run/learning, GET /pipeline/runs (history),
    GET /pipeline/runs/latest, GET /pipeline/runs/{id}, GET /pipeline/stats
  agents.py — GET /agents (list, filter by subsystem/classification), GET /agents/{id}, GET /agents/{id}/health
  stores.py — GET /stores/fragments/stats, GET /stores/fragments (query), GET /stores/fragments/{id},
    GET /stores/beliefs/stats, GET /stores/beliefs (query), GET /stores/beliefs/{id}
  config.py — GET /config/watchlist
Server entry point (providence/api/server.py) — build_state() wires stores+agents+runner, uvicorn launcher
  CLI: python -m providence.api.server [--host] [--port] [--data-dir] [--skip-perception] [--skip-adaptive]
All routes prefixed with /api/v1, OpenAPI docs at /docs
Unit tests: tests/unit/api/ (test_health, test_pipeline, test_agents, test_stores) — guarded with pytest.importorskip("fastapi")
Dependencies added: fastapi>=0.110, uvicorn[standard]>=0.27

### Phase 12: Hosting & Deployment (Session 34)
Dockerfile upgraded — multi-stage build (api + runner targets, non-root user, health checks)
  API target: FastAPI server on port 8000, HEALTHCHECK via /api/v1/health/live
  Runner target: CLI pipeline runner, HEALTHCHECK via health command
docker-compose.yml upgraded — 5 services:
  api (always running), runner (continuous, depends on api), run-once (manual profile),
  learning (learning profile), nginx (production profile with TLS)
  Shared providence-data volume, providence-net bridge network
deploy/nginx/nginx.conf — production reverse proxy:
  HTTP→HTTPS redirect, TLS 1.2/1.3, security headers (HSTS, X-Frame-Options, X-Content-Type-Options)
  Rate limiting: 30 req/s general API, 2 req/min pipeline triggers, 300s timeout for pipeline runs
  Upstream keepalive to API server
.github/workflows/ci.yml — full CI/CD pipeline:
  Jobs: lint (ruff) → test-unit (pytest+coverage) → test-integration → docker build → deploy
  Docker build verifies API health with curl after container start
  Deploy (main branch only): push to registry, SSH deploy, post-deploy health check
  Required secrets: REGISTRY_URL/USERNAME/PASSWORD, DEPLOY_HOST/USER/SSH_KEY
scripts/monitor.py — health monitoring (stdlib only, zero dependencies):
  6 checks: liveness, readiness, system health, pipeline stats, fragment/belief store stats
  Structured JSON line output for log aggregation (ELK, CloudWatch)
  --interval N for continuous polling, --verbose for detailed stats
Makefile expanded: api, api-dev, monitor, monitor-loop, docker-build, docker-api, docker-full, docker-prod
deploy/README.md — deployment guide (quick start, Docker, production, CI/CD, architecture, monitoring)

### Phase 13: Security Audit & Hardening (Session 35)
Comprehensive security audit performed — 18 findings across 7 categories:
  CRITICAL: 3 (live credentials in .env, no API auth, credential leak in errors)
  HIGH: 5 (no rate limiting, CORS wildcard, Docker .env inclusion)
  MEDIUM: 7 (request size, query injection, nginx version, input validation)
  LOW: 3 (prompt injection risk, validation error detail, payload exposure)
Security module (providence/api/security.py):
  require_api_key() — X-API-Key header validation, PROVIDENCE_API_KEY env var
  RateLimitMiddleware — sliding window per-IP (100 GET/min, 5 POST/min)
  RequestSizeLimitMiddleware — 10 MB max, 413 on exceed
  sanitize_error() — strips API keys, Bearer tokens, URLs with creds, AWS keys, DB strings
  Health/docs paths exempt from auth and rate limiting
App hardening (providence/api/app.py):
  CORS: restricted methods (GET/POST/OPTIONS), restricted headers, no wildcard in production
  Auth middleware: enabled when PROVIDENCE_API_KEY set, exempt paths for health probes
  Exception handlers: sanitized errors, no detail leakage in responses
  enable_auth parameter for testing without auth
Route hardening:
  pipeline.py — sanitized error messages, logged internally, no exc detail in HTTP response
  stores.py — no input echo in 400/404 responses
  agents.py — subsystem/classification enum validation, no agent_id echo in 404
Deployment hardening:
  .dockerignore — comprehensive exclusions (secrets, tests, docs, IDE, node_modules)
  nginx.conf — server_tokens off (hide version)
  .env.example — PROVIDENCE_API_KEY and PROVIDENCE_ENV variables documented
Security tests (tests/unit/api/test_security.py):
  TestSanitizeError — 7 tests (Anthropic key, Bearer token, API key param, URL creds, AWS key, safe msg, traceback)
  TestAgentInputValidation — 4 tests (valid/invalid subsystem, valid/invalid classification)
  TestErrorResponseSafety — 3 tests (no ID leakage in 404s, no enum leakage in 400s)
  TestCORSConfiguration — 1 test (production mode blocks unknown origins)

## Coding Standards
- Python 3.12+ with type hints everywhere
- Pydantic v2 for all schemas (frozen=True for immutable objects)
- pytest with 80%+ coverage target
- structlog for JSON-structured logging
- httpx for async HTTP clients
- SHA-256 content hashing with deterministic serialization (sorted keys)
- All prompts version-controlled in providence/prompts/
- Tests mirror source structure in tests/

## Critical Rules (Non-Negotiable)
1. Frozen = zero LLM calls. No exceptions.
2. Agent independence. Research agents never see each other's beliefs.
3. EXEC-CAPTURE supremacy. Trailing stop overrides everything including COGNIT-EXIT.
4. Max 3 trim stages. Then mandatory close. trim_pct applies to remaining, not original.
5. Data immutability. Append-only. No in-place updates. Content-hashed.
6. Machine-evaluable invalidation. Specific metrics, thresholds, operators. No vague prose.
7. No live learning. All retraining offline. Shadow mode before live deployment.
8. THESIS-RENEW interaction. COGNIT-EXIT defers CLOSE if renewal pending AND asymmetry > 0.5.
