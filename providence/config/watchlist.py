"""Ticker watchlist configuration for Providence pipeline.

The watchlist defines which securities to run perception and analysis on.
Loaded from YAML config or environment, with defaults for testing.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WatchlistEntry:
    """A single ticker in the watchlist."""
    ticker: str
    sector: str  # GICS sector name
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    tags: tuple[str, ...] = ()  # e.g. ("mega_cap", "tech", "sp500")


@dataclass(frozen=True) 
class Watchlist:
    """Immutable watchlist of tickers to trade."""
    entries: tuple[WatchlistEntry, ...]
    name: str = "default"
    max_positions: int = 20
    
    @property
    def tickers(self) -> list[str]:
        """Active tickers sorted by priority."""
        return [
            entry.ticker
            for entry in sorted(
                self.enabled_entries,
                key=lambda e: (e.priority, e.ticker)
            )
        ]
    
    @property
    def enabled_entries(self) -> list[WatchlistEntry]:
        """Only enabled entries."""
        return [entry for entry in self.entries if entry.enabled]
    
    def by_sector(self) -> dict[str, list[WatchlistEntry]]:
        """Group entries by sector."""
        result: dict[str, list[WatchlistEntry]] = {}
        for entry in self.enabled_entries:
            if entry.sector not in result:
                result[entry.sector] = []
            result[entry.sector].append(entry)
        return result
    
    def by_priority(self, priority: int) -> list[WatchlistEntry]:
        """Filter by priority level."""
        return [
            entry for entry in self.enabled_entries
            if entry.priority == priority
        ]
    
    @classmethod
    def from_yaml(cls, path: Path) -> Watchlist:
        """Load from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Watchlist:
        """Load from dict (e.g. parsed YAML)."""
        entries = []
        tickers = data.get("tickers", [])
        
        for ticker_data in tickers:
            if isinstance(ticker_data, str):
                # Simple string ticker — use defaults
                entry = WatchlistEntry(
                    ticker=ticker_data,
                    sector="",
                    enabled=True,
                    priority=1,
                    tags=()
                )
            else:
                # Dict with full config
                tags = ticker_data.get("tags", [])
                entry = WatchlistEntry(
                    ticker=ticker_data["ticker"],
                    sector=ticker_data.get("sector", ""),
                    enabled=ticker_data.get("enabled", True),
                    priority=ticker_data.get("priority", 1),
                    tags=tuple(tags) if tags else ()
                )
            entries.append(entry)
        
        return cls(
            entries=tuple(entries),
            name=data.get("name", "default"),
            max_positions=data.get("max_positions", 20)
        )
    
    @classmethod
    def default(cls) -> Watchlist:
        """Default watchlist for testing — 10 liquid, well-known tickers."""
        entries = [
            WatchlistEntry(ticker="AAPL", sector="Information Technology", priority=1, tags=("mega_cap", "tech", "sp500")),
            WatchlistEntry(ticker="MSFT", sector="Information Technology", priority=1, tags=("mega_cap", "tech", "sp500")),
            WatchlistEntry(ticker="GOOGL", sector="Communication Services", priority=1, tags=("mega_cap", "tech", "sp500")),
            WatchlistEntry(ticker="AMZN", sector="Consumer Discretionary", priority=1, tags=("mega_cap", "sp500")),
            WatchlistEntry(ticker="NVDA", sector="Information Technology", priority=1, tags=("mega_cap", "tech", "sp500")),
            WatchlistEntry(ticker="JPM", sector="Financials", priority=1, tags=("mega_cap", "finance", "sp500")),
            WatchlistEntry(ticker="JNJ", sector="Health Care", priority=1, tags=("mega_cap", "healthcare", "sp500")),
            WatchlistEntry(ticker="XOM", sector="Energy", priority=1, tags=("mega_cap", "energy", "sp500")),
            WatchlistEntry(ticker="PG", sector="Consumer Staples", priority=1, tags=("mega_cap", "consumer", "sp500")),
            WatchlistEntry(ticker="SPY", sector="ETF", priority=2, tags=("etf", "broad_market")),
        ]
        return cls(entries=tuple(entries), name="default", max_positions=20)
