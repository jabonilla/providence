# Providence — AI-Native Hedge Fund

## Current Status
<!-- Update this after each session -->
Phase: 6 IN PROGRESS (Storage layer)
Last completed session: 26 (Storage layer — FragmentStore, BeliefStore, RunStore)
Implemented components: MarketStateFragment, BeliefObject, all enums, BaseAgent ABC, AgentContext, HealthStatus, exception hierarchy, hashing utils, structured logging, PricePayload, PolygonClient, PERCEPT-PRICE, FilingPayload, EdgarClient, filing parser, PERCEPT-FILING, AgentConfig, AgentConfigRegistry, token estimation, ContextService (7-step assembly), agents.yaml config, AnthropicClient, LLMClient Protocol, response_parser, CognitFundamental agent, prompt templates v1.0 + v1.1, sample fixtures (5 stocks), integration tests, health check script, API key redaction utility, NewsPayload, OptionsPayload, CdsPayload, MacroYieldPayload, MacroEconomicPayload, FredClient, PERCEPT-NEWS, PERCEPT-OPTIONS, PERCEPT-CDS, PERCEPT-MACRO, TechnicalSignals dataclass, technical indicator functions (SMA/EMA/RSI/MACD/Bollinger/momentum), compute_all_signals aggregator, CognitTechnical agent (FROZEN — zero LLM), signal-to-belief generation, technical invalidation conditions, CognitMacro agent (ADAPTIVE — Claude Sonnet 4), cognit_macro_v1.0 prompt template, macro test fixtures, CognitEvent agent (ADAPTIVE — Claude Sonnet 4), cognit_event_v1.0 prompt template, event test fixtures, CognitNarrative agent (ADAPTIVE — Claude Sonnet 4, peer comparison), cognit_narrative_v1.0 prompt template, narrative test fixtures, CognitCrossSec agent (ADAPTIVE — Claude Sonnet 4, peer comparison), cognit_crosssec_v1.0 prompt template, crosssec test fixtures, cognition __init__.py exports (6 agents), StatisticalRegime enum (4 states), SystemRiskMode enum (4 modes), RegimeStateObject schema, SectorRegimeOverlay schema, NarrativeRegimeOverlay schema, RegimeFeatures dataclass, regime feature extraction (realized vol, drawdown, macro), HMM model (pure Python, 4-state, forward algorithm), RegimeStat agent (FROZEN — zero LLM, HMM regime classification), sector_features module (ticker→GICS mapping, per-sector feature extraction, relative stress), RegimeSector agent (FROZEN — zero LLM, per-sector HMM classification), RegimeNarr agent (ADAPTIVE — Claude Sonnet 4, narrative regime classification), regime_narr_v1.0 prompt template, narrative regime test fixtures, parse_narrative_response parser, RegimeMismatch agent (FROZEN — zero LLM, stat-narrative-sector divergence detection, risk mode escalation), regime __init__.py exports (4 agents), Decision schemas (ContributingThesis, ConflictResolution, ActiveInvalidation, SynthesizedPositionIntent, SynthesisOutput, PortfolioMetadata, ProposedPosition, PositionProposal), Action enum, DecideSynth agent (ADAPTIVE — Claude Sonnet 4, belief synthesis with conflict resolution), decide_synth_v1.0 prompt template, parse_synthesis_response parser, synthesis test fixtures, DecideOptim agent (FROZEN — zero LLM, Black-Litterman portfolio optimization with regime-aware constraints), decision __init__.py exports (2 agents), Execution schemas (ValidationResult, ValidatedProposal, RoutedOrder, RoutingPlan, GuardianCheck, GuardianVerdict, TrailingStopState, CaptureDecision, CaptureOutput), ExecValidate agent (FROZEN — pre-trade validation with risk-mode constraints), ExecRouter agent (FROZEN — order routing with strategy/urgency/slippage), ExecGuardian agent (FROZEN — kill switch and circuit breakers), ExecCapture agent (FROZEN — trailing stop management with supremacy, 3-stage trim, hard giveback), execution __init__.py exports (4 agents), Exit schemas (ExitAssessment, ExitOutput, MonitoredCondition, InvalidationMonitorOutput, RenewalCandidate, ThesisRenewalOutput, ShadowExitSignal, ShadowExitOutput, BeliefHealthReport, RenewalMonitorOutput), CognitExit agent (ADAPTIVE — Claude Sonnet 4, exit assessment with renewal deferral), parse_exit_response parser, exit test fixtures, InvalidMon agent (FROZEN — invalidation condition monitoring with breach magnitude/velocity/confidence impact), ThesisRenew agent (FROZEN — thesis renewal with confidence decay, asymmetry scoring, regime alignment), ShadowExit agent (FROZEN — shadow exit tracking, COGNIT-EXIT vs EXEC-CAPTURE divergence detection), RenewMon agent (FROZEN — renewal monitoring with belief health scoring, urgency classification), exit __init__.py exports (5 agents), Learning schemas (AgentAttribution, TickerAttribution, AttributionOutput, CalibrationBucket, AgentCalibration, CalibrationOutput, RetrainRecommendation, RetrainOutput, BacktestPeriod, BacktestOutput), LearnAttrib agent (FROZEN — offline performance attribution with hit rate, IR, Sharpe contribution), LearnCalib agent (FROZEN — offline confidence calibration with Brier score, bucket analysis), LearnRetrain agent (FROZEN — offline retraining recommendations with priority levels, shadow mode enforcement), LearnBacktest agent (FROZEN — offline backtesting with sub-period analysis, annualized metrics, profit factor), learning __init__.py exports (4 agents), Governance enums (CapitalTier, MaturityStage, IncidentSeverity), Governance schemas (TierConstraints, CapitalTierOutput, AgentMaturityRecord, MaturityGateOutput, GovernanceIncident, SystemHealthSummary, OversightOutput, PolicyViolation, PolicyOutput), GovernCapital agent (FROZEN — AUM tier classification with execution constraints), GovernMaturity agent (FROZEN — agent deployment stage evaluation with promotion criteria), GovernOversight agent (FROZEN — system health aggregation with incident detection), GovernPolicy agent (FROZEN — policy enforcement with violation detection), governance __init__.py exports (4 agents), Orchestration models (StageResult, PipelineRun, StageStatus, RunStatus), PipelineStage (isolated async executor with timeout/error isolation), Orchestrator (DAG coordinator with 4 loop methods: main/exit/learning/governance), ProvidenceRunner (scheduler with run_once/run_continuous/run_learning_batch), OrchestrationError exception, Agent factory (build_agent_registry with dependency injection, skip_perception/skip_adaptive/agent_filter), CLI entry point (__main__.py with run-once/run-continuous/run-learning/health/list-agents commands), FragmentStore (append-only, in-memory + JSONL persistence, indexed by data_type/entity), BeliefStore (append-only, indexed by agent_id/ticker), RunStore (pipeline run history with success_rate)

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
