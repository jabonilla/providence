"""Tests for Watchlist at providence/config/watchlist.py."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from providence.config.watchlist import Watchlist, WatchlistEntry


class TestWatchlistEntry:
    """Test WatchlistEntry dataclass."""

    def test_entry_creation(self):
        """WatchlistEntry can be created with required fields."""
        entry = WatchlistEntry(ticker="AAPL", sector="Information Technology")
        assert entry.ticker == "AAPL"
        assert entry.sector == "Information Technology"
        assert entry.enabled is True
        assert entry.priority == 1
        assert entry.tags == ()

    def test_entry_defaults(self):
        """WatchlistEntry has correct defaults."""
        entry = WatchlistEntry(ticker="AAPL", sector="Tech")
        assert entry.enabled is True
        assert entry.priority == 1
        assert entry.tags == ()

    def test_entry_frozen(self):
        """WatchlistEntry is frozen (immutable)."""
        entry = WatchlistEntry(ticker="AAPL", sector="Tech")
        with pytest.raises(Exception):  # FrozenInstanceError
            entry.ticker = "MSFT"

    def test_entry_with_tags(self):
        """WatchlistEntry can have tags."""
        entry = WatchlistEntry(
            ticker="AAPL",
            sector="Tech",
            tags=("mega_cap", "sp500")
        )
        assert "mega_cap" in entry.tags
        assert "sp500" in entry.tags

    def test_entry_with_priority(self):
        """WatchlistEntry can have custom priority."""
        entry = WatchlistEntry(
            ticker="AAPL",
            sector="Tech",
            priority=2
        )
        assert entry.priority == 2


class TestWatchlistDefault:
    """Test Watchlist.default() factory."""

    def test_default_returns_watchlist(self):
        """default() returns a Watchlist instance."""
        watchlist = Watchlist.default()
        assert isinstance(watchlist, Watchlist)

    def test_default_has_10_tickers(self):
        """default() has 10 tickers."""
        watchlist = Watchlist.default()
        assert len(watchlist.entries) == 10

    def test_default_all_have_sectors(self):
        """All default tickers have sectors."""
        watchlist = Watchlist.default()
        for entry in watchlist.entries:
            assert entry.sector  # Non-empty sector

    def test_default_all_enabled(self):
        """All default tickers are enabled."""
        watchlist = Watchlist.default()
        for entry in watchlist.entries:
            assert entry.enabled is True

    def test_default_includes_known_tickers(self):
        """default() includes well-known tickers."""
        watchlist = Watchlist.default()
        tickers = {entry.ticker for entry in watchlist.entries}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers


class TestWatchlistFromDict:
    """Test Watchlist.from_dict() loading."""

    def test_from_dict_simple_tickers(self):
        """from_dict handles simple string tickers."""
        data = {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "name": "test_watchlist",
        }
        watchlist = Watchlist.from_dict(data)
        assert len(watchlist.entries) == 3
        assert watchlist.tickers == ["AAPL", "GOOGL", "MSFT"]  # Sorted

    def test_from_dict_dict_format(self):
        """from_dict handles dict format with full config."""
        data = {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "sector": "Information Technology",
                    "priority": 1,
                },
                {
                    "ticker": "MSFT",
                    "sector": "Information Technology",
                    "priority": 2,
                },
            ]
        }
        watchlist = Watchlist.from_dict(data)
        assert len(watchlist.entries) == 2
        assert watchlist.entries[0].sector == "Information Technology"

    def test_from_dict_disabled_entries_kept(self):
        """from_dict includes disabled entries."""
        data = {
            "tickers": [
                {"ticker": "AAPL", "sector": "Tech", "enabled": True},
                {"ticker": "MSFT", "sector": "Tech", "enabled": False},
            ]
        }
        watchlist = Watchlist.from_dict(data)
        assert len(watchlist.entries) == 2
        assert len(watchlist.enabled_entries) == 1

    def test_from_dict_with_tags(self):
        """from_dict preserves tags."""
        data = {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "sector": "Tech",
                    "tags": ["mega_cap", "sp500"],
                }
            ]
        }
        watchlist = Watchlist.from_dict(data)
        assert "mega_cap" in watchlist.entries[0].tags


class TestWatchlistFromYaml:
    """Test Watchlist.from_yaml() loading."""

    def test_from_yaml_simple(self):
        """from_yaml loads from YAML file."""
        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "watchlist.yaml"
            yaml_content = {
                "name": "test_watchlist",
                "tickers": ["AAPL", "MSFT", "GOOGL"],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            watchlist = Watchlist.from_yaml(yaml_path)
            assert len(watchlist.entries) == 3

    def test_from_yaml_with_config(self):
        """from_yaml loads full ticker config."""
        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "watchlist.yaml"
            yaml_content = {
                "name": "test_watchlist",
                "max_positions": 15,
                "tickers": [
                    {
                        "ticker": "AAPL",
                        "sector": "Information Technology",
                        "priority": 1,
                        "enabled": True,
                    },
                ],
            }
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            watchlist = Watchlist.from_yaml(yaml_path)
            assert watchlist.max_positions == 15
            assert watchlist.entries[0].priority == 1


class TestWatchlistQueries:
    """Test Watchlist query methods."""

    @pytest.fixture
    def watchlist(self):
        """Create a test watchlist."""
        entries = [
            WatchlistEntry(
                ticker="AAPL",
                sector="Information Technology",
                priority=1,
                enabled=True,
            ),
            WatchlistEntry(
                ticker="MSFT",
                sector="Information Technology",
                priority=1,
                enabled=True,
            ),
            WatchlistEntry(
                ticker="JPM",
                sector="Financials",
                priority=2,
                enabled=True,
            ),
            WatchlistEntry(
                ticker="INACTIVE",
                sector="Tech",
                priority=1,
                enabled=False,
            ),
        ]
        return Watchlist(entries=tuple(entries), name="test")

    def test_tickers_property(self, watchlist):
        """tickers property returns enabled tickers sorted by priority."""
        tickers = watchlist.tickers
        assert len(tickers) == 3  # Only enabled
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "JPM" in tickers
        assert "INACTIVE" not in tickers

    def test_by_sector(self, watchlist):
        """by_sector groups entries by sector."""
        by_sector = watchlist.by_sector()
        assert "Information Technology" in by_sector
        assert len(by_sector["Information Technology"]) == 2
        assert "Financials" in by_sector
        assert len(by_sector["Financials"]) == 1

    def test_by_priority(self, watchlist):
        """by_priority filters by priority level."""
        priority_1 = watchlist.by_priority(1)
        assert len(priority_1) == 2
        assert all(e.priority == 1 for e in priority_1)

        priority_2 = watchlist.by_priority(2)
        assert len(priority_2) == 1

    def test_enabled_entries(self, watchlist):
        """enabled_entries returns only enabled entries."""
        enabled = watchlist.enabled_entries
        assert len(enabled) == 3
        assert all(e.enabled for e in enabled)


class TestWatchlistMaxPositions:
    """Test max_positions field."""

    def test_max_positions_default(self):
        """max_positions defaults to 20."""
        watchlist = Watchlist(entries=())
        assert watchlist.max_positions == 20

    def test_max_positions_custom(self):
        """max_positions can be customized."""
        watchlist = Watchlist(entries=(), max_positions=10)
        assert watchlist.max_positions == 10

    def test_max_positions_from_dict(self):
        """from_dict reads max_positions."""
        data = {
            "tickers": [],
            "max_positions": 15,
        }
        watchlist = Watchlist.from_dict(data)
        assert watchlist.max_positions == 15
