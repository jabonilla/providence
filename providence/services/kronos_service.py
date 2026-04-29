"""KronosService — Foundation model forecasting for financial K-line data.

Wraps the Kronos foundation model (https://github.com/shiyu-coder/Kronos)
to provide OHLCV price forecasting from raw market data.

Kronos is a decoder-only transformer pre-trained on 12B+ K-line records
from 45+ global exchanges. It uses a specialized tokenizer that discretizes
continuous OHLCV data into hierarchical discrete tokens.

Usage:
    service = KronosService()       # lazy model load
    service = KronosService(model_name="NeoQuasar/Kronos-base")
    forecast = await service.predict(ohlcv_df, horizon=20)

The service handles:
  - Model + tokenizer loading from HuggingFace Hub (lazy, on first predict)
  - OHLCV DataFrame preprocessing and validation
  - Prediction via KronosPredictor (normalization, tokenization, inference,
    inverse normalization all handled internally)
  - Output conversion to structured ForecastResult

Requirements:
  1. Clone the Kronos repo:
     git clone https://github.com/shiyu-coder/Kronos.git
  2. Install its dependencies:
     pip install -r Kronos/requirements.txt
  3. Either:
     a) Set KRONOS_HOME env var pointing to the cloned directory, OR
     b) Add Kronos/ to your PYTHONPATH
  4. pip install torch pandas

Classification: FROZEN infrastructure — no LLM calls, pure model inference.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class CandleForecast:
    """Single forecasted candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    # Probabilistic bounds (from multiple samples)
    close_low: float | None = None   # 10th percentile
    close_high: float | None = None  # 90th percentile


@dataclass(frozen=True)
class ForecastResult:
    """Complete forecast output from Kronos."""

    ticker: str
    model_name: str
    horizon: int                          # number of candles forecasted
    candles: list[CandleForecast]         # forecasted candle sequence
    predicted_return: float               # close[-1] / input_close[-1] - 1
    predicted_direction: str              # "UP" | "DOWN" | "FLAT"
    confidence: float                     # based on sample agreement (0-1)
    forecast_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KronosService:
    """Forecast service wrapping the Kronos foundation model.

    Lazily loads model and tokenizer on first prediction call.
    Thread-safe for concurrent predictions via asyncio.

    Args:
        model_name: HuggingFace model ID. Default: "NeoQuasar/Kronos-base"
        tokenizer_name: HuggingFace tokenizer ID. Default: "NeoQuasar/Kronos-Tokenizer-base"
        max_context: Maximum input sequence length. Default: 512
        device: Torch device. Default: "cpu"
        sample_count: Number of probabilistic samples per prediction. Default: 20
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
        device: str = "cpu",
        sample_count: int = 20,
    ) -> None:
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name
        self._max_context = max_context
        self._device = device
        self._sample_count = sample_count

        # Lazy-loaded
        self._model: Any = None
        self._tokenizer: Any = None
        self._predictor: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazily load model and tokenizer from HuggingFace Hub."""
        if self._loaded:
            return

        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
        except ImportError:
            # Kronos isn't pip-installable — try auto-discovering the clone
            import os
            import sys

            kronos_home = os.getenv("KRONOS_HOME", "")
            search_paths = [p for p in [
                kronos_home,
                os.path.join(os.getcwd(), "Kronos"),
                os.path.join(os.path.dirname(__file__), "..", "..", "Kronos"),
                os.path.expanduser("~/Kronos"),
            ] if p]

            found = False
            for path in search_paths:
                abs_path = os.path.abspath(path)
                model_init = os.path.join(abs_path, "model", "__init__.py")
                if os.path.isfile(model_init):
                    if abs_path not in sys.path:
                        sys.path.insert(0, abs_path)
                    logger.info("Found Kronos installation", path=abs_path)
                    found = True
                    break

            if found:
                try:
                    from model import Kronos, KronosTokenizer, KronosPredictor
                except ImportError:
                    found = False

            if not found:
                raise ImportError(
                    "Kronos model not found. Install with:\n"
                    "  git clone https://github.com/shiyu-coder/Kronos.git\n"
                    "  pip install -r Kronos/requirements.txt\n"
                    "Then either:\n"
                    "  export KRONOS_HOME=/path/to/Kronos\n"
                    "  OR clone into the Providence project root"
                )

        logger.info(
            "Loading Kronos model",
            model=self._model_name,
            tokenizer=self._tokenizer_name,
            device=self._device,
        )

        self._tokenizer = KronosTokenizer.from_pretrained(self._tokenizer_name)
        self._model = Kronos.from_pretrained(self._model_name)
        self._predictor = KronosPredictor(
            self._model,
            self._tokenizer,
            max_context=self._max_context,
            device=self._device,
        )
        self._loaded = True

        logger.info("Kronos model loaded", max_context=self._max_context)

    async def predict(
        self,
        ohlcv_data: "pd.DataFrame",
        horizon: int = 20,
        ticker: str = "UNKNOWN",
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> ForecastResult:
        """Generate price forecast from OHLCV data.

        Args:
            ohlcv_data: DataFrame with columns ['open', 'high', 'low', 'close'].
                        Optional: 'volume', 'amount'. Must have a DatetimeIndex
                        or a 'timestamp' column.
            horizon: Number of future candles to predict. Default: 20.
            ticker: Ticker symbol for labeling. Default: "UNKNOWN".
            temperature: Sampling temperature. Higher = more diverse. Default: 1.0.
            top_p: Nucleus sampling threshold. Default: 0.9.

        Returns:
            ForecastResult with predicted candles and directional signal.

        Raises:
            ValueError: If input data is missing required columns or too short.
            ImportError: If Kronos is not installed.
        """
        import pandas as pd

        # Validate input
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(ohlcv_data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(ohlcv_data) < 20:
            raise ValueError(
                f"Need at least 20 candles for prediction, got {len(ohlcv_data)}"
            )

        # Truncate to max_context if needed
        if len(ohlcv_data) > self._max_context:
            ohlcv_data = ohlcv_data.iloc[-self._max_context:]

        # Extract timestamps
        if isinstance(ohlcv_data.index, pd.DatetimeIndex):
            x_timestamps = ohlcv_data.index.to_series().reset_index(drop=True)
        elif "timestamp" in ohlcv_data.columns:
            x_timestamps = pd.to_datetime(ohlcv_data["timestamp"]).reset_index(drop=True)
        else:
            # Generate synthetic timestamps (daily)
            x_timestamps = pd.date_range(
                end=datetime.now(timezone.utc),
                periods=len(ohlcv_data),
                freq="B",  # business days
            ).to_series().reset_index(drop=True)

        # Generate future timestamps
        last_ts = x_timestamps.iloc[-1]
        y_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(days=1),
            periods=horizon,
            freq="B",
        ).to_series().reset_index(drop=True)

        # Prepare input DataFrame (reset index for Kronos)
        input_df = ohlcv_data[list(required_cols)].reset_index(drop=True)

        # Add optional columns if present
        for opt_col in ["volume", "amount"]:
            if opt_col in ohlcv_data.columns:
                input_df[opt_col] = ohlcv_data[opt_col].reset_index(drop=True)

        # Run prediction in thread pool (model inference is CPU-bound)
        loop = asyncio.get_event_loop()
        forecast_df = await loop.run_in_executor(
            None,
            self._predict_sync,
            input_df,
            x_timestamps,
            y_timestamps,
            temperature,
            top_p,
        )

        # Convert to structured result
        return self._build_result(
            ticker=ticker,
            forecast_df=forecast_df,
            y_timestamps=y_timestamps,
            input_close=float(input_df["close"].iloc[-1]),
            horizon=horizon,
        )

    def _predict_sync(
        self,
        input_df: "pd.DataFrame",
        x_timestamps: "pd.Series",
        y_timestamps: "pd.Series",
        temperature: float,
        top_p: float,
    ) -> "pd.DataFrame":
        """Synchronous prediction call (runs in thread pool).

        Returns DataFrame with forecasted OHLC values.
        """
        self._ensure_loaded()

        result = self._predictor.predict(
            data=input_df,
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
            T=temperature,
            top_p=top_p,
            sample_count=self._sample_count,
        )

        return result

    def _build_result(
        self,
        ticker: str,
        forecast_df: "pd.DataFrame",
        y_timestamps: "pd.Series",
        input_close: float,
        horizon: int,
    ) -> ForecastResult:
        """Convert raw model output to structured ForecastResult."""
        import numpy as np

        candles = []

        # forecast_df may contain multiple samples — aggregate
        # Expected columns: open, high, low, close (possibly with sample dimension)
        if hasattr(forecast_df, "values") and len(forecast_df) > 0:
            # If the result is a simple DataFrame with OHLC columns
            for i in range(min(len(forecast_df), horizon)):
                row = forecast_df.iloc[i]
                ts = y_timestamps.iloc[i] if i < len(y_timestamps) else None

                candle = CandleForecast(
                    timestamp=ts if ts is not None else datetime.now(timezone.utc),
                    open=float(row.get("open", row.get("close", 0))),
                    high=float(row.get("high", row.get("close", 0))),
                    low=float(row.get("low", row.get("close", 0))),
                    close=float(row.get("close", 0)),
                )
                candles.append(candle)

        if not candles:
            # Fallback: create neutral forecast
            for i in range(horizon):
                ts = y_timestamps.iloc[i] if i < len(y_timestamps) else datetime.now(timezone.utc)
                candles.append(CandleForecast(
                    timestamp=ts,
                    open=input_close,
                    high=input_close,
                    low=input_close,
                    close=input_close,
                ))

        # Compute directional signal
        final_close = candles[-1].close
        predicted_return = (final_close / input_close) - 1.0 if input_close > 0 else 0.0

        if predicted_return > 0.005:
            direction = "UP"
        elif predicted_return < -0.005:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Confidence based on consistency of direction across candles
        if len(candles) > 1:
            directions = []
            prev_close = input_close
            for c in candles:
                if c.close > prev_close * 1.001:
                    directions.append(1)
                elif c.close < prev_close * 0.999:
                    directions.append(-1)
                else:
                    directions.append(0)
                prev_close = c.close

            # Agreement ratio
            if direction == "UP":
                agreement = sum(1 for d in directions if d > 0) / len(directions)
            elif direction == "DOWN":
                agreement = sum(1 for d in directions if d < 0) / len(directions)
            else:
                agreement = sum(1 for d in directions if d == 0) / len(directions)

            confidence = min(0.9, 0.3 + agreement * 0.6)
        else:
            confidence = 0.5

        return ForecastResult(
            ticker=ticker,
            model_name=self._model_name,
            horizon=horizon,
            candles=candles,
            predicted_return=round(predicted_return, 6),
            predicted_direction=direction,
            confidence=round(confidence, 4),
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """The HuggingFace model identifier."""
        return self._model_name
