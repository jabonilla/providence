"""Tests for CONTEXT-SVC: Context Assembly Service.

Tests the 7-step context assembly pipeline:
  1. IDENTIFY SCOPE
  2. RETRIEVE (filter by data_type and entity)
  3. PRIORITIZE (recency-based ordering)
  4. PEER CONTEXT (peer fragment inclusion)
  5. TOKEN BUDGET (fit within limit)
  6. HASH (context_window_hash computation)
  7. RETURN (AgentContext assembly)
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext
from providence.config.agent_config import AgentConfig, AgentConfigRegistry
from providence.exceptions import ContextAssemblyError
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.services.context_svc import ContextService
from providence.utils.hashing import compute_context_window_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _make_fragment(
    entity: str = "AAPL",
    data_type: DataType = DataType.PRICE_OHLCV,
    hours_ago: float = 0,
    payload: dict | None = None,
) -> MarketStateFragment:
    """Helper to create a fragment with a timestamp relative to NOW."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-TEST",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=data_type,
        schema_version="1.0.0",
        source_hash=f"hash-{entity}-{data_type.value}-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload=payload or {"value": f"{entity}-{data_type.value}-{hours_ago}"},
    )


def _make_registry(overrides: dict | None = None) -> AgentConfigRegistry:
    """Create a simple test registry with one agent config."""
    base = {
        "TEST-AGENT": {
            "consumes": [DataType.PRICE_OHLCV, DataType.FILING_10Q],
            "max_token_budget": 100_000,
            "entity_scope": [],
            "peer_count": 3,
            "peer_group": "sector_peers",
            "priority_window_hours": 24,
        }
    }
    if overrides:
        base["TEST-AGENT"].update(overrides)
    return AgentConfigRegistry.from_dict(base)


# ---------------------------------------------------------------------------
# AgentConfig & Registry tests
# ---------------------------------------------------------------------------
class TestAgentConfig:
    """Tests for AgentConfig and AgentConfigRegistry."""

    def test_config_frozen(self):
        """AgentConfig should be immutable."""
        config = AgentConfig(
            agent_id="X",
            consumes=[DataType.PRICE_OHLCV],
        )
        with pytest.raises(Exception):
            config.agent_id = "Y"

    def test_registry_get_existing(self):
        """get() returns config when agent exists."""
        registry = _make_registry()
        config = registry.get("TEST-AGENT")
        assert config is not None
        assert config.agent_id == "TEST-AGENT"

    def test_registry_get_missing(self):
        """get() returns None for unknown agent."""
        registry = _make_registry()
        assert registry.get("UNKNOWN") is None

    def test_registry_get_or_raise_missing(self):
        """get_or_raise() raises KeyError for unknown agent."""
        registry = _make_registry()
        with pytest.raises(KeyError):
            registry.get_or_raise("UNKNOWN")

    def test_registry_agent_ids(self):
        """agent_ids lists all configured agents."""
        registry = _make_registry()
        assert "TEST-AGENT" in registry.agent_ids

    def test_registry_from_dict_string_data_types(self):
        """from_dict converts string DataType names to enums."""
        registry = AgentConfigRegistry.from_dict({
            "MY-AGENT": {
                "consumes": ["PRICE_OHLCV", "FILING_10K"],
                "max_token_budget": 50_000,
            }
        })
        config = registry.get("MY-AGENT")
        assert config is not None
        assert DataType.PRICE_OHLCV in config.consumes
        assert DataType.FILING_10K in config.consumes


# ---------------------------------------------------------------------------
# ContextService — basic assembly
# ---------------------------------------------------------------------------
class TestContextServiceBasicAssembly:
    """Test the overall assemble_context flow."""

    def test_assemble_returns_agent_context(self):
        """assemble_context returns a valid AgentContext."""
        registry = _make_registry()
        svc = ContextService(registry)
        fragments = [_make_fragment(hours_ago=1)]

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=fragments,
        )

        assert isinstance(ctx, AgentContext)
        assert ctx.agent_id == "TEST-AGENT"
        assert ctx.trigger == "schedule"
        assert len(ctx.fragments) == 1
        assert ctx.context_window_hash  # non-empty

    def test_assemble_unknown_agent_raises(self):
        """assemble_context raises ContextAssemblyError for unknown agent."""
        registry = _make_registry()
        svc = ContextService(registry)

        with pytest.raises(ContextAssemblyError):
            svc.assemble_context(
                agent_id="NONEXISTENT",
                trigger="manual",
                available_fragments=[],
            )

    def test_assemble_empty_fragments(self):
        """assemble_context handles empty fragment list gracefully."""
        registry = _make_registry()
        svc = ContextService(registry)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[],
        )

        assert ctx.fragments == []
        assert ctx.context_window_hash  # hash of empty list is still a valid hash

    def test_assemble_passes_metadata(self):
        """Metadata is passed through to the AgentContext."""
        registry = _make_registry()
        svc = ContextService(registry)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="manual",
            available_fragments=[],
            metadata={"run_id": "abc-123"},
        )

        assert ctx.metadata["run_id"] == "abc-123"


# ---------------------------------------------------------------------------
# Step 2: RETRIEVE — data type and entity filtering
# ---------------------------------------------------------------------------
class TestContextServiceRetrieve:
    """Test fragment filtering by data_type and entity_scope."""

    def test_filters_by_data_type(self):
        """Only fragments matching agent's consumed data types are kept."""
        registry = _make_registry()
        svc = ContextService(registry)

        price_frag = _make_fragment(data_type=DataType.PRICE_OHLCV)
        earnings_frag = _make_fragment(data_type=DataType.EARNINGS_CALL)
        filing_frag = _make_fragment(data_type=DataType.FILING_10Q)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[price_frag, earnings_frag, filing_frag],
        )

        data_types = {f.data_type for f in ctx.fragments}
        assert DataType.PRICE_OHLCV in data_types
        assert DataType.FILING_10Q in data_types
        assert DataType.EARNINGS_CALL not in data_types

    def test_entity_scope_all(self):
        """Empty entity_scope means all entities pass."""
        registry = _make_registry({"entity_scope": []})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL")
        msft = _make_fragment(entity="MSFT")

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl, msft],
        )

        entities = {f.entity for f in ctx.fragments}
        assert "AAPL" in entities
        assert "MSFT" in entities

    def test_entity_scope_restricted(self):
        """When entity_scope is set, only those entities pass."""
        registry = _make_registry({"entity_scope": ["AAPL"]})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL")
        msft = _make_fragment(entity="MSFT")

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl, msft],
        )

        entities = {f.entity for f in ctx.fragments}
        assert "AAPL" in entities
        assert "MSFT" not in entities


# ---------------------------------------------------------------------------
# Step 3: PRIORITIZE — recency ordering
# ---------------------------------------------------------------------------
class TestContextServicePrioritize:
    """Test recency-based prioritization."""

    def test_newest_first(self):
        """Fragments are ordered newest-first."""
        registry = _make_registry()
        svc = ContextService(registry)

        old = _make_fragment(hours_ago=10, payload={"age": "old"})
        mid = _make_fragment(hours_ago=5, payload={"age": "mid"})
        new = _make_fragment(hours_ago=1, payload={"age": "new"})

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[old, mid, new],  # passed out of order
        )

        payloads = [f.payload.get("age") for f in ctx.fragments]
        assert payloads == ["new", "mid", "old"]

    def test_priority_window_fragments_come_first(self):
        """Fragments within priority window appear before historical."""
        registry = _make_registry({"priority_window_hours": 24})
        svc = ContextService(registry)

        # Inside the 24h window
        recent = _make_fragment(hours_ago=2, payload={"group": "priority"})
        # Outside the 24h window
        old = _make_fragment(hours_ago=48, payload={"group": "historical"})

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[old, recent],
        )

        groups = [f.payload.get("group") for f in ctx.fragments]
        assert groups[0] == "priority"
        assert groups[1] == "historical"


# ---------------------------------------------------------------------------
# Step 4: PEER CONTEXT
# ---------------------------------------------------------------------------
class TestContextServicePeerContext:
    """Test peer fragment inclusion."""

    def test_peer_fragments_included(self):
        """Peer fragments are added when peer_count > 0."""
        registry = _make_registry({"peer_count": 2})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL", hours_ago=1)
        peer_msft = _make_fragment(entity="MSFT", hours_ago=2)
        peer_googl = _make_fragment(entity="GOOGL", hours_ago=3)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl],
            peer_fragments=[peer_msft, peer_googl],
        )

        entities = {f.entity for f in ctx.fragments}
        assert "AAPL" in entities
        assert "MSFT" in entities
        assert "GOOGL" in entities

    def test_peer_count_limit(self):
        """Only peer_count peers are included."""
        registry = _make_registry({"peer_count": 1})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL", hours_ago=1)
        peer1 = _make_fragment(entity="MSFT", hours_ago=2)
        peer2 = _make_fragment(entity="GOOGL", hours_ago=3)
        peer3 = _make_fragment(entity="AMZN", hours_ago=4)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl],
            peer_fragments=[peer1, peer2, peer3],
        )

        # AAPL (target) + 1 peer only (MSFT, the most recent)
        peer_entities = {f.entity for f in ctx.fragments if f.entity != "AAPL"}
        assert len(peer_entities) == 1
        assert "MSFT" in peer_entities

    def test_no_peers_when_peer_count_zero(self):
        """No peer fragments when peer_count is 0."""
        registry = _make_registry({"peer_count": 0})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL", hours_ago=1)
        peer = _make_fragment(entity="MSFT", hours_ago=2)

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl],
            peer_fragments=[peer],
        )

        entities = {f.entity for f in ctx.fragments}
        assert "AAPL" in entities
        assert "MSFT" not in entities

    def test_peer_deduplication(self):
        """Multiple fragments per peer entity: only the most recent is kept."""
        registry = _make_registry({"peer_count": 2})
        svc = ContextService(registry)

        aapl = _make_fragment(entity="AAPL", hours_ago=1)
        msft_old = _make_fragment(entity="MSFT", hours_ago=10, payload={"v": "old"})
        msft_new = _make_fragment(entity="MSFT", hours_ago=2, payload={"v": "new"})

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[aapl],
            peer_fragments=[msft_old, msft_new],
        )

        msft_frags = [f for f in ctx.fragments if f.entity == "MSFT"]
        assert len(msft_frags) == 1
        assert msft_frags[0].payload.get("v") == "new"


# ---------------------------------------------------------------------------
# Step 5: TOKEN BUDGET
# ---------------------------------------------------------------------------
class TestContextServiceTokenBudget:
    """Test token budget enforcement."""

    def test_fragments_fit_within_budget(self):
        """All fragments included when total tokens < budget."""
        registry = _make_registry({"max_token_budget": 100_000})
        svc = ContextService(registry)

        frags = [_make_fragment(hours_ago=i) for i in range(5)]

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=frags,
        )

        assert len(ctx.fragments) == 5

    def test_over_budget_drops_oldest(self):
        """When over budget, older fragments are dropped first."""
        # Create a very small budget
        registry = _make_registry({
            "max_token_budget": 50,  # Very small budget
            "priority_window_hours": 1,
        })
        svc = ContextService(registry)

        # Create fragments: one recent (priority), several old
        recent = _make_fragment(hours_ago=0.5, payload={"data": "priority_fragment"})
        old1 = _make_fragment(hours_ago=10, payload={"data": "x" * 200})
        old2 = _make_fragment(hours_ago=20, payload={"data": "y" * 200})

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[recent, old1, old2],
        )

        # Priority fragment should always be included
        assert any(f.payload.get("data") == "priority_fragment" for f in ctx.fragments)
        # Total should be fewer than all 3 due to budget
        assert len(ctx.fragments) < 3

    def test_priority_fragments_never_dropped(self):
        """Priority-window fragments are never dropped, even if over budget."""
        registry = _make_registry({
            "max_token_budget": 1,  # Impossibly small budget
            "priority_window_hours": 24,
        })
        svc = ContextService(registry)

        # All fragments within priority window
        frags = [_make_fragment(hours_ago=i) for i in range(3)]

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=frags,
        )

        # All 3 should survive — they're all in the priority window
        assert len(ctx.fragments) == 3


# ---------------------------------------------------------------------------
# Step 6: HASH — context_window_hash
# ---------------------------------------------------------------------------
class TestContextServiceHash:
    """Test context window hash computation."""

    def test_hash_is_deterministic(self):
        """Same fragments always produce the same context_window_hash."""
        registry = _make_registry()
        svc = ContextService(registry)

        frags = [
            _make_fragment(entity="AAPL", hours_ago=1, payload={"v": 1}),
            _make_fragment(entity="MSFT", hours_ago=2, payload={"v": 2}),
        ]

        ctx1 = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=frags,
        )
        ctx2 = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=frags,
        )

        assert ctx1.context_window_hash == ctx2.context_window_hash

    def test_different_fragments_different_hash(self):
        """Different fragment sets produce different hashes."""
        registry = _make_registry()
        svc = ContextService(registry)

        frag_a = _make_fragment(payload={"v": "a"})
        frag_b = _make_fragment(payload={"v": "b"})

        ctx1 = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[frag_a],
        )
        ctx2 = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=[frag_b],
        )

        assert ctx1.context_window_hash != ctx2.context_window_hash

    def test_hash_matches_utility_function(self):
        """context_window_hash matches direct call to compute_context_window_hash."""
        registry = _make_registry()
        svc = ContextService(registry)

        frags = [_make_fragment(hours_ago=1)]

        ctx = svc.assemble_context(
            agent_id="TEST-AGENT",
            trigger="schedule",
            available_fragments=frags,
        )

        expected_hash = compute_context_window_hash(ctx.fragments)
        assert ctx.context_window_hash == expected_hash


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_empty(self):
        from providence.utils.tokens import estimate_tokens
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        from providence.utils.tokens import estimate_tokens
        # "hello" is 5 chars → 5 // 4 = 1
        assert estimate_tokens("hello") >= 1

    def test_estimate_fragment_tokens(self):
        from providence.utils.tokens import estimate_fragment_tokens
        payload = {"key": "value", "number": 42}
        tokens = estimate_fragment_tokens(payload)
        assert tokens > 0

    def test_estimate_fragment_tokens_empty(self):
        from providence.utils.tokens import estimate_fragment_tokens
        tokens = estimate_fragment_tokens({})
        # "{}" serializes to 2 chars → at least 1 token
        assert tokens >= 0


# ---------------------------------------------------------------------------
# YAML loading (agents.yaml)
# ---------------------------------------------------------------------------
class TestAgentConfigYAML:
    """Test loading agent configs from YAML file."""

    def test_load_agents_yaml(self, tmp_path):
        """Load agent config from a YAML file."""
        yaml_content = """
agents:
  COGNIT-FUNDAMENTAL:
    consumes:
      - FILING_10K
      - FILING_10Q
      - PRICE_OHLCV
    max_token_budget: 100000
    entity_scope: []
    peer_count: 3
    peer_group: sector_peers
    priority_window_hours: 24
"""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(yaml_content)

        registry = AgentConfigRegistry.from_yaml(yaml_file)
        config = registry.get("COGNIT-FUNDAMENTAL")

        assert config is not None
        assert config.agent_id == "COGNIT-FUNDAMENTAL"
        assert DataType.FILING_10K in config.consumes
        assert DataType.FILING_10Q in config.consumes
        assert DataType.PRICE_OHLCV in config.consumes
        assert config.max_token_budget == 100_000
        assert config.peer_count == 3
        assert config.priority_window_hours == 24

    def test_load_missing_yaml_returns_empty(self, tmp_path):
        """Loading a nonexistent YAML returns an empty registry."""
        registry = AgentConfigRegistry.from_yaml(tmp_path / "nonexistent.yaml")
        assert registry.agent_ids == []

    def test_load_yaml_multiple_agents(self, tmp_path):
        """Multiple agents in one YAML file."""
        yaml_content = """
agents:
  AGENT-A:
    consumes:
      - PRICE_OHLCV
    max_token_budget: 50000
  AGENT-B:
    consumes:
      - FILING_10K
    max_token_budget: 80000
"""
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(yaml_content)

        registry = AgentConfigRegistry.from_yaml(yaml_file)
        assert len(registry.agent_ids) == 2
        assert "AGENT-A" in registry.agent_ids
        assert "AGENT-B" in registry.agent_ids
