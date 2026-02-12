"""Response parser for Research Agent LLM outputs.

Parses raw JSON dict from LLM into validated BeliefObject.
Handles partial/malformed responses gracefully.

Spec Reference: Technical Spec v2.3, Section 4.2
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog

from providence.schemas.belief import (
    Belief,
    BeliefMetadata,
    BeliefObject,
    EvidenceRef,
    InvalidationCondition,
)
from providence.schemas.enums import (
    CatalystType,
    ComparisonOperator,
    ConditionStatus,
    Direction,
    Magnitude,
    MarketCapBucket,
)

logger = structlog.get_logger()


def parse_llm_response(
    raw: dict[str, Any],
    agent_id: str,
    context_window_hash: str,
    available_fragment_ids: set[UUID] | None = None,
) -> BeliefObject | None:
    """Parse a raw LLM JSON response into a validated BeliefObject.

    Args:
        raw: Parsed JSON dict from LLM response.
        agent_id: The agent that produced this response.
        context_window_hash: Hash of the input context window.
        available_fragment_ids: Set of fragment IDs in the context window
            for evidence ref validation.

    Returns:
        Validated BeliefObject, or None if parsing fails.
    """
    try:
        beliefs_raw = raw.get("beliefs", [])
        if not beliefs_raw:
            logger.error("LLM response contains no beliefs", agent_id=agent_id)
            return None

        beliefs: list[Belief] = []
        for b in beliefs_raw:
            belief = _parse_belief(b, available_fragment_ids)
            if belief is not None:
                # Spec requires at least 2 machine-evaluable invalidation conditions
                if len(belief.invalidation_conditions) < 2:
                    logger.warning(
                        "Dropping belief with fewer than 2 invalidation conditions",
                        thesis_id=belief.thesis_id,
                        condition_count=len(belief.invalidation_conditions),
                    )
                    continue
                beliefs.append(belief)

        if not beliefs:
            logger.error(
                "No valid beliefs parsed from LLM response",
                agent_id=agent_id,
                raw_count=len(beliefs_raw),
            )
            return None

        return BeliefObject(
            belief_id=uuid4(),
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc),
            context_window_hash=context_window_hash,
            beliefs=beliefs,
        )

    except Exception as e:
        logger.error(
            "Failed to parse LLM response into BeliefObject",
            agent_id=agent_id,
            error=str(e),
        )
        return None


def _parse_belief(
    raw: dict[str, Any],
    available_fragment_ids: set[UUID] | None = None,
) -> Belief | None:
    """Parse a single belief from LLM JSON."""
    try:
        # Parse direction
        direction_str = raw.get("direction", "").upper()
        direction = Direction(direction_str)

        # Parse magnitude
        magnitude_str = raw.get("magnitude", "").upper()
        magnitude = Magnitude(magnitude_str)

        # Parse confidence (clamp to [0, 1])
        raw_confidence = float(raw.get("raw_confidence", 0.5))
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Parse time horizon
        time_horizon_days = int(raw.get("time_horizon_days", 90))

        # Parse evidence refs
        evidence = _parse_evidence_refs(
            raw.get("evidence", []),
            available_fragment_ids,
        )

        # Parse invalidation conditions
        invalidation_conditions = _parse_invalidation_conditions(
            raw.get("invalidation_conditions", [])
        )

        # Parse metadata
        metadata = _parse_metadata(raw.get("metadata", {}))

        # Parse correlated beliefs
        correlated = raw.get("correlated_beliefs", [])

        return Belief(
            thesis_id=raw.get("thesis_id", f"THESIS-{uuid4().hex[:8].upper()}"),
            ticker=raw.get("ticker", "UNKNOWN"),
            thesis_summary=raw.get("thesis_summary", "No summary provided"),
            direction=direction,
            magnitude=magnitude,
            raw_confidence=raw_confidence,
            time_horizon_days=time_horizon_days,
            evidence=evidence,
            invalidation_conditions=invalidation_conditions,
            correlated_beliefs=correlated,
            metadata=metadata,
        )

    except (ValueError, KeyError) as e:
        logger.warning(
            "Failed to parse individual belief",
            error=str(e),
            thesis_id=raw.get("thesis_id"),
        )
        return None


def _parse_evidence_refs(
    raw_refs: list[dict[str, Any]],
    available_fragment_ids: set[UUID] | None = None,
) -> list[EvidenceRef]:
    """Parse evidence references, validating fragment IDs if possible."""
    refs: list[EvidenceRef] = []

    for r in raw_refs:
        try:
            frag_id_str = r.get("source_fragment_id", "")
            try:
                frag_id = UUID(frag_id_str)
            except (ValueError, AttributeError):
                # LLM may produce non-UUID fragment IDs — skip
                logger.debug(
                    "Skipping evidence ref with invalid UUID",
                    raw_id=frag_id_str,
                )
                continue

            # Validate the fragment ID exists in context if we have that info
            if available_fragment_ids and frag_id not in available_fragment_ids:
                logger.debug(
                    "Evidence ref points to fragment not in context",
                    fragment_id=str(frag_id),
                )
                # Still include it — the LLM may reference valid IDs

            weight = float(r.get("weight", 0.5))
            weight = max(0.0, min(1.0, weight))

            refs.append(
                EvidenceRef(
                    source_fragment_id=frag_id,
                    field_path=r.get("field_path", "payload"),
                    observation=r.get("observation", ""),
                    weight=weight,
                )
            )
        except Exception as e:
            logger.debug("Skipping malformed evidence ref", error=str(e))

    return refs


def _parse_invalidation_conditions(
    raw_conditions: list[dict[str, Any]],
) -> list[InvalidationCondition]:
    """Parse invalidation conditions, enforcing machine-evaluability."""
    conditions: list[InvalidationCondition] = []

    for c in raw_conditions:
        try:
            metric = c.get("metric", "")
            if not metric:
                logger.debug(
                    "Skipping invalidation condition with empty metric",
                    description=c.get("description"),
                )
                continue

            # Parse operator
            operator_str = c.get("operator", "").upper()
            # Handle common LLM variations
            operator_map = {
                ">": "GT",
                "<": "LT",
                "=": "EQ",
                "==": "EQ",
                "GREATER_THAN": "GT",
                "LESS_THAN": "LT",
                "EQUAL": "EQ",
            }
            operator_str = operator_map.get(operator_str, operator_str)

            try:
                operator = ComparisonOperator(operator_str)
            except ValueError:
                logger.debug(
                    "Skipping condition with invalid operator",
                    operator=operator_str,
                )
                continue

            threshold = c.get("threshold")
            if threshold is None:
                logger.debug(
                    "Skipping condition with missing threshold",
                    metric=metric,
                )
                continue

            conditions.append(
                InvalidationCondition(
                    condition_id=uuid4(),
                    description=c.get("description", f"{metric} {operator_str} {threshold}"),
                    data_source_agent=c.get("data_source_agent", "PERCEPT-PRICE"),
                    metric=metric,
                    operator=operator,
                    threshold=float(threshold),
                    status=ConditionStatus.ACTIVE,
                    current_value=c.get("current_value"),
                    breach_magnitude=c.get("breach_magnitude"),
                    breach_velocity=c.get("breach_velocity"),
                )
            )
        except Exception as e:
            logger.debug(
                "Skipping malformed invalidation condition",
                error=str(e),
            )

    return conditions


def _parse_metadata(raw: dict[str, Any]) -> BeliefMetadata:
    """Parse belief metadata with safe defaults."""
    try:
        sector = raw.get("sector", "Unknown")

        # Parse market cap bucket
        bucket_str = raw.get("market_cap_bucket", "LARGE").upper()
        try:
            market_cap_bucket = MarketCapBucket(bucket_str)
        except ValueError:
            market_cap_bucket = MarketCapBucket.LARGE

        # Parse catalyst type
        catalyst_str = raw.get("catalyst_type", "NONE").upper()
        try:
            catalyst_type = CatalystType(catalyst_str)
        except ValueError:
            catalyst_type = CatalystType.NONE

        return BeliefMetadata(
            sector=sector,
            market_cap_bucket=market_cap_bucket,
            catalyst_type=catalyst_type,
        )
    except Exception:
        return BeliefMetadata(
            sector="Unknown",
            market_cap_bucket=MarketCapBucket.LARGE,
            catalyst_type=CatalystType.NONE,
        )
