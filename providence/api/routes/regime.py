"""Regime state endpoints — sector overlays and global regime status."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state
from providence.api.schemas import (
    RegimeStateResponse,
    SectorRegimeOverlayResponse,
)

router = APIRouter(prefix="/regime", tags=["regime"])


def _extract_regime_state(run) -> dict | None:
    """Extract regime state from a PipelineRun's metadata or stage results.

    Regime data lives in two places:
    1. metadata["regime_state"] — written by REGIME-MISMATCH (the final regime agent)
    2. stage_results where agent_id contains "REGIME-SECTOR" — per-sector overlays

    Returns a dict with the regime state or None if not found.
    """
    # Try metadata first (most complete — comes from REGIME-MISMATCH)
    regime_state = run.metadata.get("regime_state")
    if regime_state:
        if isinstance(regime_state, dict):
            return regime_state
        # If it's a Pydantic model, dump it
        if hasattr(regime_state, "model_dump"):
            return regime_state.model_dump()

    # Fallback: try regime_outputs in metadata
    regime_outputs = run.metadata.get("regime_outputs", {})
    if regime_outputs:
        # Look for REGIME-MISMATCH first, then any regime agent
        for agent_key in ["REGIME-MISMATCH", "regime_mismatch", "regime-mismatch"]:
            if agent_key in regime_outputs:
                output = regime_outputs[agent_key]
                if isinstance(output, dict):
                    return output
                if hasattr(output, "model_dump"):
                    return output.model_dump()

        # Return the first regime output found
        for _key, output in regime_outputs.items():
            if isinstance(output, dict):
                return output
            if hasattr(output, "model_dump"):
                return output.model_dump()

    # Fallback: extract from stage results
    for sr in run.stage_results:
        if "regime" in sr.agent_id.lower() and "mismatch" in sr.agent_id.lower():
            if sr.output:
                return sr.output
        if "regime" in sr.agent_id.lower() and "sector" in sr.agent_id.lower():
            if sr.output:
                return sr.output

    return None


@router.get("/state", response_model=RegimeStateResponse)
async def get_regime_state() -> RegimeStateResponse:
    """Get the current regime state from the latest pipeline run.

    Returns global statistical regime, narrative overlay, risk mode,
    and sector-level overlays.
    """
    state = get_state()
    if state.run_store is None:
        raise HTTPException(status_code=503, detail="Run store not initialized")

    # Get the latest MAIN loop run
    latest_run = state.run_store.get_latest(loop_type="MAIN")
    if latest_run is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found")

    regime_data = _extract_regime_state(latest_run)
    if regime_data is None:
        raise HTTPException(status_code=404, detail="No regime data in latest run")

    # Build sector overlays
    sector_overlays = []
    raw_overlays = regime_data.get("sector_overlays", {})
    for sector_name, overlay in raw_overlays.items():
        if isinstance(overlay, dict):
            sector_overlays.append(
                SectorRegimeOverlayResponse(
                    sector=overlay.get("sector", sector_name),
                    regime=overlay.get("regime", "TRANSITION_UNCERTAIN"),
                    regime_confidence=overlay.get("regime_confidence", 0.0),
                    regime_probabilities=overlay.get("regime_probabilities", {}),
                    relative_stress=overlay.get("relative_stress", 0.0),
                    key_signals=overlay.get("key_signals", []),
                    ticker_count=overlay.get("ticker_count", 0),
                )
            )

    # Build narrative overlay
    narrative = regime_data.get("narrative_overlay")
    narrative_label = None
    narrative_confidence = None
    narrative_key_signals = []
    narrative_affected_sectors = []
    narrative_summary = None
    if isinstance(narrative, dict):
        narrative_label = narrative.get("label")
        narrative_confidence = narrative.get("confidence")
        narrative_key_signals = narrative.get("key_signals", [])
        narrative_affected_sectors = narrative.get("affected_sectors", [])
        narrative_summary = narrative.get("summary")

    return RegimeStateResponse(
        statistical_regime=regime_data.get("statistical_regime", "TRANSITION_UNCERTAIN"),
        regime_confidence=regime_data.get("regime_confidence", 0.0),
        regime_probabilities=regime_data.get("regime_probabilities", {}),
        system_risk_mode=regime_data.get("system_risk_mode", "NORMAL"),
        sector_overlays=sector_overlays,
        narrative_label=narrative_label,
        narrative_confidence=narrative_confidence,
        narrative_key_signals=narrative_key_signals,
        narrative_affected_sectors=narrative_affected_sectors,
        narrative_summary=narrative_summary,
        run_id=str(latest_run.run_id),
        timestamp=latest_run.started_at.isoformat(),
    )


@router.get("/sectors", response_model=list[SectorRegimeOverlayResponse])
async def get_sector_overlays() -> list[SectorRegimeOverlayResponse]:
    """Get sector-level regime overlays from the latest pipeline run.

    Returns per-sector regime classifications with confidence,
    stress levels, and driving signals. Used by the portal's
    regime breakdown view.
    """
    state = get_state()
    if state.run_store is None:
        raise HTTPException(status_code=503, detail="Run store not initialized")

    # Get the latest MAIN loop run
    latest_run = state.run_store.get_latest(loop_type="MAIN")
    if latest_run is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found")

    regime_data = _extract_regime_state(latest_run)
    if regime_data is None:
        raise HTTPException(status_code=404, detail="No regime data in latest run")

    raw_overlays = regime_data.get("sector_overlays", {})
    results = []
    for sector_name, overlay in raw_overlays.items():
        if isinstance(overlay, dict):
            results.append(
                SectorRegimeOverlayResponse(
                    sector=overlay.get("sector", sector_name),
                    regime=overlay.get("regime", "TRANSITION_UNCERTAIN"),
                    regime_confidence=overlay.get("regime_confidence", 0.0),
                    regime_probabilities=overlay.get("regime_probabilities", {}),
                    relative_stress=overlay.get("relative_stress", 0.0),
                    key_signals=overlay.get("key_signals", []),
                    ticker_count=overlay.get("ticker_count", 0),
                )
            )

    return results
