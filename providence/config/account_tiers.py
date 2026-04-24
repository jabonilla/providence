"""Account tier definitions and limits for Providence.

Defines four tiers (EXPLORER, INVESTOR, PRO, FUND) with
feature gates and resource limits for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccountTier(str, Enum):
    """User account tier levels."""

    EXPLORER = "EXPLORER"      # Free tier
    INVESTOR = "INVESTOR"      # Basic paid
    PRO = "PRO"                # Professional
    FUND = "FUND"              # Institutional


@dataclass(frozen=True)
class TierLimits:
    """Resource limits and feature flags for an account tier."""

    max_agents: int
    max_positions: int
    custom_weights: bool
    custom_config: bool
    api_access: bool
    max_watchlist: int
    shadow_mode: bool
    paper_trading: bool
    live_trading: bool
    doc_uploads_per_day: int


TIER_LIMITS: dict[AccountTier, TierLimits] = {
    AccountTier.EXPLORER: TierLimits(
        max_agents=3,
        max_positions=5,
        custom_weights=False,
        custom_config=False,
        api_access=False,
        max_watchlist=10,
        shadow_mode=True,
        paper_trading=False,
        live_trading=False,
        doc_uploads_per_day=3,
    ),
    AccountTier.INVESTOR: TierLimits(
        max_agents=6,
        max_positions=15,
        custom_weights=True,
        custom_config=False,
        api_access=True,
        max_watchlist=25,
        shadow_mode=True,
        paper_trading=True,
        live_trading=False,
        doc_uploads_per_day=10,
    ),
    AccountTier.PRO: TierLimits(
        max_agents=6,
        max_positions=30,
        custom_weights=True,
        custom_config=True,
        api_access=True,
        max_watchlist=50,
        shadow_mode=True,
        paper_trading=True,
        live_trading=True,
        doc_uploads_per_day=50,
    ),
    AccountTier.FUND: TierLimits(
        max_agents=35,
        max_positions=100,
        custom_weights=True,
        custom_config=True,
        api_access=True,
        max_watchlist=200,
        shadow_mode=True,
        paper_trading=True,
        live_trading=True,
        doc_uploads_per_day=500,
    ),
}


def get_tier_limits(tier: AccountTier) -> TierLimits:
    """Get the resource limits for a given account tier.

    Parameters
    ----------
    tier:
        The account tier to look up.

    Returns
    -------
    TierLimits:
        The limits and feature flags for that tier.

    Raises
    ------
    KeyError:
        If the tier is not found (should not happen with valid enum values).
    """
    return TIER_LIMITS[tier]
