# Providence Frontend Assessment
## Backend ↔ Frontend Integration Gap Analysis

---

## 1. Executive Summary

The Providence backend is **100% spec-complete** as an autonomous hedge fund engine: 35 agents, 7 subsystems, orchestration layer, portfolio tracker, 1525+ tests passing. However, the backend was built as a **CLI-driven autonomous system** — it has no REST API, no authentication, no WebSocket server, and no user/investor data model.

The frontend spec expects **13+ REST endpoints, JWT authentication, 4 WebSocket events, and multi-tenant investor scoping**. Before the frontend can connect to real data, a full **API layer** must be built.

**The good news:** The backend's Pydantic schemas are the exact data contracts the frontend needs. Every field the frontend renders has a corresponding schema field in the backend. The mapping is clean.

**Recommended approach:** Start the frontend with mock data (Sessions 1-8), build the API layer as a parallel workstream, then wire them together in Session 9 — exactly as the session prompts suggest.

---

## 2. What Exists (Backend)

| Component | Status | Details |
|-----------|--------|---------|
| Agent pipeline | ✅ Complete | 35 agents across 7 subsystems |
| Pydantic schemas | ✅ Complete | ~90 frozen models with content hashing |
| Orchestration | ✅ Complete | DAG coordinator, 4 loop types |
| Portfolio tracker | ✅ Complete | Real-time P&L, exposure tracking |
| Storage layer | ✅ Complete | Append-only JSONL (fragments, beliefs, runs) |
| External APIs | ✅ Complete | Polygon, EDGAR, FRED, Alpaca, Anthropic |
| Configuration | ✅ Complete | YAML-based, per-agent configs |
| CLI interface | ✅ Complete | run-once, run-continuous, health, list-agents |
| REST API | ❌ Missing | No FastAPI/Flask/Django |
| Authentication | ❌ Missing | No JWT, no user sessions |
| WebSocket server | ❌ Missing | No real-time event streaming |
| User/Investor model | ❌ Missing | No multi-tenant data model |
| Database (SQL) | ❌ Missing | JSONL storage only, no PostgreSQL |

---

## 3. What the Frontend Expects

### 3.1 REST Endpoints Required

| Endpoint | Method | Backend Schema Source |
|----------|--------|---------------------|
| `/api/portfolio/summary?period=` | GET | `PortfolioSnapshot` + `PortfolioMetadata` |
| `/api/performance/timeseries?period=` | GET | Computed from `PortfolioTracker` history |
| `/api/positions?status=&sector=&sort=` | GET | `Position` from tracker + `BeliefObject` for thesis |
| `/api/positions/:ticker` | GET | `Position` + `BeliefObject` + `CaptureDecision` + `ThesisRenewalOutput` |
| `/api/agents/summary` | GET | `AgentScorecard` + `AgentCalibration` + health checks |
| `/api/beliefs/recent?limit=` | GET | `BeliefObject` from `BeliefStore` |
| `/api/decisions?limit=&ticker=` | GET | `SynthesisOutput` + `ConflictResolution` |
| `/api/regime/current` | GET | `RegimeStateObject` + `SectorRegimeOverlay` + `NarrativeRegimeOverlay` |
| `/api/regime/history?period=` | GET | Historical `RegimeStateObject` records |
| `/api/system/status` | GET | `SystemHealthSummary` + subsystem health |
| `/api/reports` | GET | Generated PDF reports (new capability) |
| `/api/audit?page=&limit=&type=&actor=` | GET | `GovernanceIncident` + `PolicyViolation` + run logs |
| `/api/shadow-exit/summary` | GET | `ShadowExitOutput` + `ShadowExitSignal` |
| `/api/auth/login` | POST | New — JWT issuance |
| `/api/auth/verify-email` | POST | New — email verification |
| `/api/access-requests` | POST | New — prospective investor form |
| `/api/reports/generate` | POST | New — PDF generation |

### 3.2 WebSocket Events Required

| Event | Trigger | Backend Source |
|-------|---------|---------------|
| `agent.belief_created` | New high-conviction belief | `BeliefStore` write hook |
| `system.status_change` | Subsystem status change | `SystemHealthSummary` delta |
| `position.status_change` | Position opened/closed/trimmed | `CaptureOutput` or `ExecutionReport` |
| `regime.transition` | Market regime change | `RegimeStateObject` transition |

### 3.3 Authentication & Authorization

| Requirement | Details |
|-------------|---------|
| JWT bearer tokens | httpOnly cookie storage |
| Login/logout | Email + password |
| Role-based access | INVESTOR vs ADMIN roles |
| Investor scoping | Backend filters data per investor |
| Password reset flow | Token-based email flow |
| Email verification | Token-based verification |
| 2FA support | TOTP-based two-factor auth |
| Session management | Active sessions list, revocation |

### 3.4 Admin Endpoints (Additional)

| Endpoint | Purpose |
|----------|---------|
| `/api/admin/investors` | CRUD investor accounts |
| `/api/admin/tiers` | Capital tier management |
| `/api/admin/agents` | Agent control (pause, shadow mode) |
| `/api/admin/system` | Kill switch, manual overrides |
| `/api/admin/governance` | Governance log |
| `/api/admin/invites` | Invite code management |

---

## 4. Schema-to-Frontend Mapping

This is the strength of the architecture. The backend schemas map directly to frontend data needs:

### Overview Page
- **4 Metric Cards** → `PortfolioSnapshot` (total_value, daily_pnl, total_return, sharpe)
- **Performance Chart** → time series of `PortfolioSnapshot` records
- **Regime Card** → `RegimeStateObject` (current regime, confidence)
- **Sector Donut** → `SectorRegimeOverlay` (sector allocations)
- **Agent Contributions** → `AgentScorecard` (per-agent P&L attribution)
- **Positions Table** → `Position` objects from `PortfolioTracker`
- **System Status** → `SystemHealthSummary` (6 subsystem statuses)

### Performance Page
- **Hero Chart** → `PortfolioSnapshot` time series vs benchmark
- **Monthly Heatmap** → Aggregated monthly returns from snapshots
- **Attribution** → `TickerAttribution` + `AgentAttribution` from learning subsystem
- **Risk Metrics** → Computed from `PortfolioSnapshot` history (Sharpe, Sortino, max drawdown, etc.)

### Positions Page
- **Position Table** → `Position` + `BeliefObject` (entry thesis)
- **Position Drawer** → `BeliefObject` (thesis text, conviction history), `CaptureDecision` (trailing stop state), `InvalidationCondition` (exit triggers), `ThesisRenewalOutput` (renewal status)

### Agent Activity Page
- **7 Agent Cards** → `AgentScorecard` per agent
- **Belief Feed** → `BeliefObject` from `BeliefStore` (recent, sorted by timestamp)
- **Shadow Exit Panel** → `ShadowExitOutput` + `ShadowExitSignal`

### Decisions Page
- **Decision Log** → `SynthesisOutput` (includes `ContributingThesis`, `ConflictResolution`)
- **Decision Detail** → `SynthesisOutput` + `PositionProposal` + `GuardianVerdict`

### Market Regime Page
- **Regime Hero** → `RegimeStateObject` (regime label, confidence, HMM states)
- **History Chart** → Historical `RegimeStateObject` transitions
- **Sector Overlays** → `SectorRegimeOverlay` grid

### Audit Log
- **Event Table** → `GovernanceIncident` + `PolicyViolation` + `PipelineRun` metadata

---

## 5. Gaps to Build

### Priority 1: API Layer (Required for Session 9 wire-up)

```
providence/api/
├── __init__.py
├── app.py              # FastAPI application factory
├── dependencies.py     # Auth dependency injection, DB sessions
├── middleware.py        # CORS, rate limiting, error handling
├── auth/
│   ├── router.py       # Login, logout, refresh, verify-email, reset-password
│   ├── jwt.py          # Token creation/validation
│   └── models.py       # User, Session, InviteCode (SQLAlchemy or similar)
├── portfolio/
│   ├── router.py       # /api/portfolio/summary, /api/performance/timeseries
│   └── service.py      # Query PortfolioTracker, compute time series
├── positions/
│   ├── router.py       # /api/positions, /api/positions/:ticker
│   └── service.py      # Query tracker + belief store + capture decisions
├── agents/
│   ├── router.py       # /api/agents/summary, /api/beliefs/recent
│   └── service.py      # Query scorecards, belief store
├── decisions/
│   ├── router.py       # /api/decisions
│   └── service.py      # Query synthesis outputs from run store
├── regime/
│   ├── router.py       # /api/regime/current, /api/regime/history
│   └── service.py      # Query regime state store
├── system/
│   ├── router.py       # /api/system/status
│   └── service.py      # Health check aggregation
├── audit/
│   ├── router.py       # /api/audit
│   └── service.py      # Query governance incidents + run logs
├── reports/
│   ├── router.py       # /api/reports, /api/reports/generate
│   └── service.py      # PDF generation from templates
├── admin/
│   ├── router.py       # All /api/admin/* endpoints
│   └── service.py      # Investor CRUD, tier mgmt, agent control
└── ws/
    ├── manager.py       # WebSocket connection manager
    └── events.py        # Event broadcasting (belief, status, position, regime)
```

### Priority 2: Database Layer

The JSONL storage works for the autonomous engine, but the API layer needs:
- **User/Investor table** — accounts, roles, hashed passwords, onboarding state
- **Session table** — JWT refresh tokens, active sessions
- **Invite table** — invite codes, usage tracking
- **Report table** — generated report metadata
- **Audit table** — queryable audit events (the current JSONL audit log is append-only)

Options: PostgreSQL with SQLAlchemy (recommended), or extend JSONL stores with query indexes.

### Priority 3: WebSocket Server

- FastAPI's native WebSocket support
- Connection manager with per-investor scoping
- Hook into storage layer write paths to emit events
- 4 event types: belief_created, status_change, position_change, regime_transition

---

## 6. Recommended Build Order

### Phase A: Frontend with Mock Data (Sessions 1-8)
Build the entire investor portal UI using mock data. This is **completely independent** of the API layer and can start immediately.

- Session 1: Project setup, design tokens, UI primitives
- Session 2: Auth pages (Login, Request Access, Forgot Password, Verification)
- Session 3: Layout shell (Sidebar, PageHeader, PageTransition)
- Session 4: Overview page (all 8 sections)
- Session 5: Performance page (hero chart, heatmap, attribution)
- Session 6: Positions page (table, filters, detail drawer)
- Session 7: Agent Activity page (agent cards, belief feed, shadow exit)
- Session 8: Decisions + Regime + Reports + Audit pages

### Phase B: API Layer (Parallel with Phase A, or after)
Build FastAPI endpoints that query existing backend stores and serve JSON.

- B1: FastAPI app scaffold + auth system (JWT, user model, PostgreSQL)
- B2: Read-only investor endpoints (portfolio, positions, agents, beliefs)
- B3: Decision + regime + audit endpoints
- B4: Admin endpoints + WebSocket server
- B5: Report generation + system status

### Phase C: Wire-Up (Session 9+)
Replace mock data with real API calls using React Query hooks.

### Phase D: Admin + Polish (Sessions 13-20)
Admin dashboard, account settings, onboarding, WebSocket integration, responsive, testing.

---

## 7. Agent Naming Alignment

The frontend spec shows 7 agents in the Agent Activity page. Mapping to backend:

| Frontend Display Name | Backend Agent ID | Notes |
|----------------------|-----------------|-------|
| Fundamental | COGNIT-FUND | ✅ Direct match |
| Technical | COGNIT-TECH | ✅ Direct match |
| Macro | COGNIT-MACRO | ✅ Direct match |
| Sentiment | COGNIT-SENT | ✅ Direct match |
| Sector | COGNIT-SECTOR | ✅ Direct match |
| Cross-Sectional | COGNIT-CROSSSEC | ✅ Direct match |
| Shadow Exit | COGNIT-EXIT | ✅ Direct match (frozen agent, no LLM) |

The frontend also shows DECIDE-SYNTH, DECIDE-OPTIM, and the 4 execution agents in the Decisions page, but the Agent Activity page focuses on the 7 cognition agents. This maps cleanly.

---

## 8. Risk Items

1. **No database migration path yet.** The JSONL stores work for the engine but won't scale for multi-investor queries. A PostgreSQL migration for at least user/auth data is needed.

2. **Investor scoping doesn't exist.** The backend is single-tenant. The API layer must add investor-level data filtering — or the initial portal can show the single fund's data to all authenticated investors (which is the simpler, more realistic approach for a hedge fund portal).

3. **Report generation is new.** The spec expects monthly PDF reports. This is a new capability that doesn't exist in the backend — it needs a template engine + PDF generator.

4. **Real-time events need storage hooks.** WebSocket events require hooks in the storage layer write paths. The current `BeliefStore.append()` and `RunStore.append()` would need event emission callbacks.

---

## 9. Bottom Line

**Start frontend immediately.** The mock data approach through Session 8 is the right call — it lets you build the entire UI without waiting for the API layer. The schema mapping is clean enough that when you wire up in Session 9, the data shapes will match.

**The API layer is ~2-3 sessions of work** (FastAPI scaffold + auth + endpoints + WebSocket). It can be built in parallel or sequentially after Session 8.

**Total scope estimate:** 20 frontend sessions + 3-5 API layer sessions = ~23-25 sessions to production.
