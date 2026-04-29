"""Kronos forecasting endpoints.

GET  /api/v1/forecast/{ticker}  — Get Kronos price forecast for a ticker
GET  /api/v1/forecast           — Get forecasts for all watchlist tickers
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from providence.api.deps import get_state

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{ticker}")
async def get_forecast(
    ticker: str,
    horizon: int = Query(default=20, ge=5, le=60, description="Forecast horizon in trading days"),
) -> dict[str, Any]:
    """Get Kronos foundation model price forecast for a single ticker.

    Uses historical PRICE_OHLCV fragments from the FragmentStore to build
    the input series, then runs Kronos model inference.

    Args:
        ticker: Ticker symbol (e.g., AAPL, MSFT).
        horizon: Number of future candles to predict. Default: 20.

    Returns:
        Forecast result with predicted candles, direction, and confidence.
    """
    state = get_state()
    ticker = ticker.upper()

    # Get historical price fragments for this ticker
    if not state.fragment_store:
        raise HTTPException(status_code=503, detail="Fragment store not available")

    from providence.schemas.enums import DataType

    fragments = state.fragment_store.query(
        data_types=[DataType.PRICE_OHLCV],
        entities=[ticker],
    )

    if not fragments or len(fragments) < 20:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient price data for forecast (need 20+, have {len(fragments) if fragments else 0})",
        )

    # Build OHLCV DataFrame from fragments
    import pandas as pd

    sorted_frags = sorted(fragments, key=lambda f: f.timestamp)
    records = []
    for frag in sorted_frags:
        payload = frag.payload
        if "close" in payload:
            records.append({
                "open": float(payload.get("open", payload["close"])),
                "high": float(payload.get("high", payload["close"])),
                "low": float(payload.get("low", payload["close"])),
                "close": float(payload["close"]),
                "volume": float(payload.get("volume", 0)),
            })

    if len(records) < 20:
        raise HTTPException(status_code=404, detail="Insufficient valid price records")

    df = pd.DataFrame(records)

    # Run Kronos forecast
    try:
        from providence.services.kronos_service import KronosService

        # Use a shared service instance if available, otherwise create one
        service = getattr(state, "_kronos_service", None)
        if service is None:
            service = KronosService()
            state._kronos_service = service  # cache for reuse

        forecast = await service.predict(
            ohlcv_data=df,
            horizon=horizon,
            ticker=ticker,
        )

        return {
            "ticker": forecast.ticker,
            "model": forecast.model_name,
            "horizon": forecast.horizon,
            "predicted_direction": forecast.predicted_direction,
            "predicted_return": forecast.predicted_return,
            "confidence": forecast.confidence,
            "forecast_timestamp": forecast.forecast_timestamp.isoformat(),
            "current_close": float(df["close"].iloc[-1]),
            "candles": [
                {
                    "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                }
                for c in forecast.candles
            ],
        }

    except ImportError:
        # Kronos model not installed — return a helpful message
        return {
            "ticker": ticker,
            "model": "NeoQuasar/Kronos-base",
            "status": "model_not_installed",
            "message": (
                "Kronos model is not installed in this environment. "
                "Install with: git clone https://github.com/shiyu-coder/Kronos.git "
                "&& pip install -r Kronos/requirements.txt "
                "&& export KRONOS_HOME=$(pwd)/Kronos"
            ),
            "horizon": horizon,
            "current_close": float(df["close"].iloc[-1]),
            "data_points_available": len(records),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Forecast generation failed")


@router.get("")
async def get_watchlist_forecasts(
    horizon: int = Query(default=20, ge=5, le=60),
) -> dict[str, Any]:
    """Get Kronos forecasts for all watchlist tickers.

    Returns a summary forecast for each ticker in the configured watchlist.
    """
    state = get_state()

    tickers = []
    if state.watchlist:
        tickers = state.watchlist.tickers
    elif state.fragment_store:
        # Fall back to tickers that have data
        from providence.schemas.enums import DataType
        all_frags = state.fragment_store.query(data_types=[DataType.PRICE_OHLCV])
        tickers = list(set(f.entity for f in all_frags if f.entity))

    if not tickers:
        return {"forecasts": [], "message": "No tickers configured"}

    results = []
    for ticker in tickers[:10]:  # Cap at 10 to avoid timeout
        try:
            forecast = await get_forecast(ticker, horizon)
            results.append(forecast)
        except HTTPException:
            results.append({
                "ticker": ticker,
                "status": "insufficient_data",
            })
        except Exception:
            results.append({
                "ticker": ticker,
                "status": "error",
            })

    return {
        "forecasts": results,
        "horizon": horizon,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
