# Providence — AI-Native Hedge Fund

## Current Status
<!-- Update this after each session -->
Phase: 2 (Multi-Agent + Synthesis)
Last completed session: 8 (Remaining Perception Agents)
Implemented components: MarketStateFragment, BeliefObject, all enums, BaseAgent ABC, AgentContext, HealthStatus, exception hierarchy, hashing utils, structured logging, PricePayload, PolygonClient, PERCEPT-PRICE, FilingPayload, EdgarClient, filing parser, PERCEPT-FILING, AgentConfig, AgentConfigRegistry, token estimation, ContextService (7-step assembly), agents.yaml config, AnthropicClient, LLMClient Protocol, response_parser, CognitFundamental agent, prompt templates v1.0 + v1.1, sample fixtures (5 stocks), integration tests, health check script, API key redaction utility, NewsPayload, OptionsPayload, CdsPayload, MacroYieldPayload, MacroEconomicPayload, FredClient, PERCEPT-NEWS, PERCEPT-OPTIONS, PERCEPT-CDS, PERCEPT-MACRO

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

### Phase 3: Exit System (Session 17)
COGNIT-EXIT + INVALID-MON + THESIS-RENEW + SHADOW-EXIT + RENEW-MON

### Phase 4: Learning + Production (Future)
Attribution, calibration, retraining, backtesting, governance

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
