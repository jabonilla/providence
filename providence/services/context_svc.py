"""CONTEXT-SVC: Context Assembly Service.

The bridge between Perception and Cognition. Assembles the input context
window for each Research Agent by selecting, prioritizing, and fitting
MarketStateFragments within a token budget.

Spec Reference: Technical Spec v2.3, Section 4.6

Classification: FROZEN — no LLM calls. Pure data selection and assembly.

Steps:
  1. IDENTIFY SCOPE  → What entities does this agent cover?
  2. RETRIEVE        → Filter relevant fragments by data_type and entity
  3. PRIORITIZE      → Recent data at full detail, historical summarized
  4. PEER CONTEXT    → Include peer data for entity-specific agents
  5. TOKEN BUDGET    → Fit within configured token limit
  6. HASH            → Compute context_window_hash
  7. RETURN          → AgentContext with assembled fragments + hash
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from providence.agents.base import AgentContext
from providence.config.agent_config import AgentConfig, AgentConfigRegistry
from providence.exceptions import ContextAssemblyError
from providence.schemas.enums import DataType
from providence.schemas.market_state import MarketStateFragment
from providence.utils.hashing import compute_context_window_hash
from providence.utils.tokens import estimate_fragment_tokens

logger = structlog.get_logger()


class ContextService:
    """Assembles context windows for Research Agents.

    Takes a pool of available MarketStateFragments and produces a
    focused, token-budget-compliant AgentContext for each agent.
    """

    def __init__(self, config_registry: AgentConfigRegistry) -> None:
        self._registry = config_registry

    def assemble_context(
        self,
        agent_id: str,
        trigger: str,
        available_fragments: list[MarketStateFragment],
        peer_fragments: list[MarketStateFragment] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Assemble a context window for the specified agent.

        Args:
            agent_id: ID of the agent to build context for.
            trigger: What triggered this run ("schedule", "event", "manual").
            available_fragments: All available MarketStateFragments.
            peer_fragments: Additional peer entity fragments (optional).
            metadata: Additional metadata to include in context.

        Returns:
            AgentContext with selected and prioritized fragments.

        Raises:
            ContextAssemblyError: If agent config not found.
        """
        config = self._registry.get(agent_id)
        if config is None:
            raise ContextAssemblyError(
                message=f"No configuration found for agent: {agent_id}",
                agent_id=agent_id,
            )

        # Step 1: IDENTIFY SCOPE
        target_data_types = set(config.consumes)

        # Step 2: RETRIEVE — filter by data_type and entity scope
        filtered = self._retrieve(available_fragments, target_data_types, config)

        # Step 3: PRIORITIZE — sort by recency, relevance
        prioritized = self._prioritize(filtered, config)

        # Step 4: PEER CONTEXT — add peer fragments if configured
        if peer_fragments and config.peer_count > 0:
            peer_selected = self._select_peers(
                peer_fragments, target_data_types, config
            )
            prioritized.extend(peer_selected)

        # Step 5: TOKEN BUDGET — fit within limit
        budgeted = self._apply_token_budget(prioritized, config)

        # Step 6: HASH
        context_hash = compute_context_window_hash(budgeted)

        # Step 7: RETURN
        return AgentContext(
            agent_id=agent_id,
            trigger=trigger,
            fragments=budgeted,
            context_window_hash=context_hash,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    def _retrieve(
        self,
        fragments: list[MarketStateFragment],
        target_data_types: set[DataType],
        config: AgentConfig,
    ) -> list[MarketStateFragment]:
        """Step 2: Filter fragments by data type and entity scope."""
        result: list[MarketStateFragment] = []

        for frag in fragments:
            # Must match one of the agent's consumed data types
            if frag.data_type not in target_data_types:
                continue

            # If entity scope is specified, filter by it
            if config.entity_scope:
                if frag.entity is not None and frag.entity not in config.entity_scope:
                    continue

            result.append(frag)

        return result

    def _prioritize(
        self,
        fragments: list[MarketStateFragment],
        config: AgentConfig,
    ) -> list[MarketStateFragment]:
        """Step 3: Sort by recency (newest first), then by relevance.

        Fragments within the priority window (last N hours) are always
        kept at the front. Older fragments are sorted by timestamp.
        """
        now = datetime.now(timezone.utc)
        priority_cutoff = now - timedelta(hours=config.priority_window_hours)

        # Split into priority (recent) and historical
        priority: list[MarketStateFragment] = []
        historical: list[MarketStateFragment] = []

        for frag in fragments:
            if frag.timestamp >= priority_cutoff:
                priority.append(frag)
            else:
                historical.append(frag)

        # Sort each group by timestamp descending (newest first)
        priority.sort(key=lambda f: f.timestamp, reverse=True)
        historical.sort(key=lambda f: f.timestamp, reverse=True)

        # Priority fragments come first, then historical
        return priority + historical

    def _select_peers(
        self,
        peer_fragments: list[MarketStateFragment],
        target_data_types: set[DataType],
        config: AgentConfig,
    ) -> list[MarketStateFragment]:
        """Step 4: Select peer entity fragments.

        Picks the most recent fragment per peer entity, up to peer_count.
        """
        # Filter by data type
        matching = [f for f in peer_fragments if f.data_type in target_data_types]

        # Group by entity, pick most recent per entity
        entity_latest: dict[str | None, MarketStateFragment] = {}
        for frag in matching:
            existing = entity_latest.get(frag.entity)
            if existing is None or frag.timestamp > existing.timestamp:
                entity_latest[frag.entity] = frag

        # Sort by timestamp (newest first) and take up to peer_count
        peers = sorted(entity_latest.values(), key=lambda f: f.timestamp, reverse=True)
        return peers[: config.peer_count]

    def _apply_token_budget(
        self,
        fragments: list[MarketStateFragment],
        config: AgentConfig,
    ) -> list[MarketStateFragment]:
        """Step 5: Fit fragments within the token budget.

        Never drops fragments from the priority window for the target entity.
        Drops oldest/least relevant fragments first when over budget.
        """
        if not fragments:
            return []

        budget = config.max_token_budget
        now = datetime.now(timezone.utc)
        priority_cutoff = now - timedelta(hours=config.priority_window_hours)

        selected: list[MarketStateFragment] = []
        tokens_used = 0

        for frag in fragments:
            frag_tokens = estimate_fragment_tokens(frag.payload)

            # Priority fragments (recent + target entity) are never dropped
            is_priority = (
                frag.timestamp >= priority_cutoff
                and (
                    not config.entity_scope
                    or frag.entity in config.entity_scope
                )
            )

            if is_priority:
                selected.append(frag)
                tokens_used += frag_tokens
            elif tokens_used + frag_tokens <= budget:
                selected.append(frag)
                tokens_used += frag_tokens
            else:
                # Over budget — skip this fragment
                logger.debug(
                    "Fragment dropped due to token budget",
                    agent_id=config.agent_id,
                    fragment_id=str(frag.fragment_id),
                    tokens_used=tokens_used,
                    budget=budget,
                )

        return selected
