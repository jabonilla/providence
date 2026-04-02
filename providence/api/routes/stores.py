"""Storage query endpoints — fragments, beliefs, and run stats."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from providence.api.deps import get_state
from providence.api.schemas import (
    BeliefDetailResponse,
    BeliefStoreStatsResponse,
    BeliefSummaryResponse,
    FragmentDetailResponse,
    FragmentStoreStatsResponse,
    FragmentSummaryResponse,
)

router = APIRouter(prefix="/stores", tags=["stores"])


# ── Fragment Store ──────────────────────────────────────────────────

@router.get("/fragments/stats", response_model=FragmentStoreStatsResponse)
async def get_fragment_stats() -> FragmentStoreStatsResponse:
    """Get fragment store statistics."""
    state = get_state()
    store = state.fragment_store

    from providence.schemas.enums import DataType, ValidationStatus

    # Count by data type using the index
    by_type = {}
    for dt in DataType:
        ids = store._by_data_type.get(dt, [])
        if ids:
            by_type[dt.value] = len(ids)

    # Count by validation status by scanning all fragments
    status_summary: dict[str, int] = {}
    for frag in store._fragments.values():
        vs = frag.validation_status.value if hasattr(frag.validation_status, 'value') else str(frag.validation_status)
        status_summary[vs] = status_summary.get(vs, 0) + 1

    return FragmentStoreStatsResponse(
        total_count=store.count(),
        by_type=by_type,
        by_validation_status=status_summary,
    )


@router.get("/fragments", response_model=list[FragmentSummaryResponse])
async def list_fragments(
    entity: Optional[str] = Query(None, description="Filter by ticker/entity"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(50, ge=1, le=500),
) -> list[FragmentSummaryResponse]:
    """List market state fragments (summary, no payload)."""
    state = get_state()

    kwargs: dict[str, Any] = {"limit": limit}
    if entity:
        kwargs["entities"] = {entity}
    if data_type:
        from providence.schemas.enums import DataType
        try:
            kwargs["data_types"] = {DataType(data_type)}
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid data_type parameter",
            )

    fragments = state.fragment_store.query(**kwargs)
    return [
        FragmentSummaryResponse(
            fragment_id=f.fragment_id,
            agent_id=f.agent_id,
            timestamp=f.timestamp,
            entity=f.entity,
            data_type=f.data_type.value if hasattr(f.data_type, 'value') else str(f.data_type),
            validation_status=f.validation_status.value if hasattr(f.validation_status, 'value') else str(f.validation_status),
            schema_version=f.schema_version,
        )
        for f in fragments
    ]


@router.get("/fragments/{fragment_id}", response_model=FragmentDetailResponse)
async def get_fragment(fragment_id: UUID) -> FragmentDetailResponse:
    """Get a single fragment with full payload."""
    state = get_state()
    fragment = state.fragment_store.get(fragment_id)
    if fragment is None:
        raise HTTPException(status_code=404, detail="Fragment not found")

    return FragmentDetailResponse(
        fragment_id=fragment.fragment_id,
        agent_id=fragment.agent_id,
        timestamp=fragment.timestamp,
        entity=fragment.entity,
        data_type=fragment.data_type.value if hasattr(fragment.data_type, 'value') else str(fragment.data_type),
        validation_status=fragment.validation_status.value if hasattr(fragment.validation_status, 'value') else str(fragment.validation_status),
        schema_version=fragment.schema_version,
        payload=fragment.payload,
    )


# ── Belief Store ────────────────────────────────────────────────────

@router.get("/beliefs/stats", response_model=BeliefStoreStatsResponse)
async def get_belief_stats() -> BeliefStoreStatsResponse:
    """Get belief store statistics."""
    state = get_state()
    store = state.belief_store
    return BeliefStoreStatsResponse(
        total_count=store.count(),
        agents=sorted(store.all_agents()),
        tickers=sorted(store.all_tickers()),
    )


@router.get("/beliefs", response_model=list[BeliefSummaryResponse])
async def list_beliefs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    limit: int = Query(50, ge=1, le=500),
) -> list[BeliefSummaryResponse]:
    """List belief objects (summary, no full beliefs array)."""
    state = get_state()

    kwargs: dict[str, Any] = {"limit": limit}
    if agent_id:
        kwargs["agent_ids"] = {agent_id}
    if ticker:
        kwargs["tickers"] = {ticker}

    beliefs = state.belief_store.query(**kwargs)
    return [
        BeliefSummaryResponse(
            belief_id=b.belief_id,
            agent_id=b.agent_id,
            timestamp=b.timestamp,
            belief_count=len(b.beliefs),
            tickers=list({belief.ticker for belief in b.beliefs}),
        )
        for b in beliefs
    ]


@router.get("/beliefs/{belief_id}", response_model=BeliefDetailResponse)
async def get_belief(belief_id: UUID) -> BeliefDetailResponse:
    """Get a single belief object with full beliefs array."""
    state = get_state()
    belief = state.belief_store.get(belief_id)
    if belief is None:
        raise HTTPException(status_code=404, detail="Belief not found")

    return BeliefDetailResponse(
        belief_id=belief.belief_id,
        agent_id=belief.agent_id,
        timestamp=belief.timestamp,
        context_window_hash=belief.context_window_hash,
        beliefs=[b.model_dump() for b in belief.beliefs],
    )
