"""Usage analytics endpoints — per-user usage tracking and tier limit checks."""

from __future__ import annotations

from fastapi import APIRouter, Query

from providence.api.deps import get_state

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/summary")
async def get_usage_summary(
    user_id: str = Query(default="default"),
    days: int = Query(default=30, ge=1, le=365),
):
    """Get aggregated usage summary for a user."""
    state = get_state()
    if state.usage_tracker is None:
        return {
            "user_id": user_id,
            "period_start": "",
            "period_end": "",
            "api_calls": 0,
            "pipeline_runs": 0,
            "signals_generated": 0,
            "doc_uploads": 0,
            "chat_messages": 0,
            "beliefs_created": 0,
            "agent_invocations": 0,
        }
    summary = state.usage_tracker.get_summary(user_id=user_id, days=days)
    return summary.model_dump()


@router.get("/daily")
async def get_daily_usage(
    user_id: str = Query(default="default"),
    days: int = Query(default=30, ge=1, le=365),
):
    """Get per-day usage breakdown."""
    state = get_state()
    if state.usage_tracker is None:
        return []
    daily = state.usage_tracker.get_daily_breakdown(user_id=user_id, days=days)
    return [d.model_dump() for d in daily]


@router.get("/breakdown")
async def get_usage_breakdown(
    user_id: str = Query(default="default"),
    days: int = Query(default=30, ge=1, le=365),
):
    """Get usage counts by event type."""
    state = get_state()
    if state.usage_tracker is None:
        return {}
    return state.usage_tracker.get_type_breakdown(user_id=user_id, days=days)


@router.get("/limits")
async def check_usage_limits(
    user_id: str = Query(default="default"),
):
    """Check current usage against tier limits."""
    state = get_state()

    # Resolve tier
    from dataclasses import asdict

    from providence.config.account_tiers import AccountTier, get_tier_limits

    tier_str = state.extra.get("account_tier", "EXPLORER")
    try:
        tier = AccountTier(tier_str)
    except ValueError:
        tier = AccountTier.EXPLORER
    limits = get_tier_limits(tier)

    result = {
        "tier": tier.value,
        "limits": asdict(limits),
        "usage_today": {},
        "allowed": {},
    }

    if state.usage_tracker is None:
        return result

    # Check doc uploads
    allowed_docs, remaining_docs = state.usage_tracker.check_limit(
        user_id, "doc_upload", limits.doc_uploads_per_day,
    )
    result["usage_today"]["doc_uploads"] = limits.doc_uploads_per_day - remaining_docs
    result["allowed"]["doc_uploads"] = allowed_docs

    # Check chat messages (100/day for all tiers for now)
    allowed_chat, remaining_chat = state.usage_tracker.check_limit(
        user_id, "chat_message", 100,
    )
    result["usage_today"]["chat_messages"] = 100 - remaining_chat
    result["allowed"]["chat_messages"] = allowed_chat

    # Check pipeline runs (varies by tier)
    max_runs = {
        "EXPLORER": 5,
        "INVESTOR": 20,
        "PRO": 100,
        "FUND": 1000,
    }.get(tier.value, 5)
    allowed_runs, remaining_runs = state.usage_tracker.check_limit(
        user_id, "pipeline_run", max_runs,
    )
    result["usage_today"]["pipeline_runs"] = max_runs - remaining_runs
    result["allowed"]["pipeline_runs"] = allowed_runs

    return result
