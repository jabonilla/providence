"""Tests for PerceptionScheduler at providence/services/perception_scheduler.py."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providence.config.watchlist import Watchlist, WatchlistEntry
from providence.services.perception_scheduler import PerceptionScheduler


class TestRunFullSweep:
    """Test run_full_sweep method."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler with mocked components."""
        agents = {
            "agent_1": AsyncMock(),
            "agent_2": AsyncMock(),
        }
        fragment_store = MagicMock()
        fragment_store.append.return_value = True
        fragment_store.append_many.return_value = 1
        watchlist = Watchlist.default()

        return PerceptionScheduler(
            perception_agents=agents,
            fragment_store=fragment_store,
            watchlist=watchlist,
            inter_ticker_delay=0.01,  # Short delay for testing
            inter_agent_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_run_full_sweep_processes_all_tickers(self, scheduler):
        """run_full_sweep processes all enabled tickers."""
        # Mock agent execution
        for agent in scheduler._agents.values():
            agent.process.return_value = {
                "ticker": "AAPL",
                "agent_id": "test",
                "fragment": "test_fragment",
            }

        result = await scheduler.run_full_sweep()

        assert result["tickers_processed"] == len(scheduler._watchlist.enabled_entries)
        assert "sweep_start" in result
        assert "sweep_end" in result

    @pytest.mark.asyncio
    async def test_run_full_sweep_returns_stats(self, scheduler):
        """run_full_sweep returns sweep statistics."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {"agent_id": "test", "fragment": "data"}

        result = await scheduler.run_full_sweep()

        assert "duration_seconds" in result
        assert "tickers_processed" in result
        assert "agents_run" in result
        assert "fragments_created" in result


class TestRunPrioritySweep:
    """Test run_priority_sweep method."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with mixed priorities."""
        agents = {
            "agent_1": AsyncMock(),
            "agent_2": AsyncMock(),
        }
        fragment_store = MagicMock()
        fragment_store.append.return_value = True
        fragment_store.append_many.return_value = 1

        # Create watchlist with different priorities
        entries = [
            WatchlistEntry(ticker="AAPL", sector="Tech", priority=1),
            WatchlistEntry(ticker="MSFT", sector="Tech", priority=1),
            WatchlistEntry(ticker="ZZZ", sector="Tech", priority=2),
        ]
        watchlist = Watchlist(entries=tuple(entries))

        return PerceptionScheduler(
            perception_agents=agents,
            fragment_store=fragment_store,
            watchlist=watchlist,
            inter_ticker_delay=0.01,
            inter_agent_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_run_priority_sweep_filters_by_priority(self, scheduler):
        """run_priority_sweep only processes specified priority."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {"agent_id": "test", "fragment": "data"}

        result = await scheduler.run_priority_sweep(max_priority=1)

        # Should only process priority-1 tickers (2 of them)
        assert result["tickers_processed"] <= 2

    @pytest.mark.asyncio
    async def test_run_priority_sweep_returns_stats(self, scheduler):
        """run_priority_sweep returns statistics."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {"agent_id": "test", "fragment": "data"}

        result = await scheduler.run_priority_sweep(max_priority=1)

        assert "tickers_processed" in result
        assert "duration_seconds" in result


class TestRunSingle:
    """Test run_single method."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        agents = {
            "agent_1": AsyncMock(),
            "agent_2": AsyncMock(),
        }
        fragment_store = MagicMock()
        fragment_store.append.return_value = True
        fragment_store.append_many.return_value = 1
        watchlist = Watchlist.default()

        return PerceptionScheduler(
            perception_agents=agents,
            fragment_store=fragment_store,
            watchlist=watchlist,
            inter_ticker_delay=0.01,
            inter_agent_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_run_single_processes_one_ticker(self, scheduler):
        """run_single processes a single ticker."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {
                "ticker": "AAPL",
                "agent_id": "test",
                "fragment": "data",
            }

        result = await scheduler.run_single("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["agents_run"] == len(scheduler._agents)

    @pytest.mark.asyncio
    async def test_run_single_returns_fragment_count(self, scheduler):
        """run_single returns fragment count."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {
                "ticker": "AAPL",
                "agent_id": "test",
                "fragment": "data",
            }

        result = await scheduler.run_single("AAPL")

        assert "fragments" in result
        assert result["fragments"] >= 0


class TestAgentFailureIsolation:
    """Test that one agent failure doesn't stop others."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with agents."""
        agents = {
            "good_agent": AsyncMock(),
            "bad_agent": AsyncMock(),
        }
        # Make bad_agent raise exception
        agents["bad_agent"].process.side_effect = Exception("Agent error")

        fragment_store = MagicMock()
        fragment_store.append.return_value = True
        fragment_store.append_many.return_value = 1
        watchlist = Watchlist.default()

        return PerceptionScheduler(
            perception_agents=agents,
            fragment_store=fragment_store,
            watchlist=watchlist,
            inter_ticker_delay=0.01,
            inter_agent_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_agent_failure_isolation(self, scheduler):
        """One agent failure doesn't stop other agents."""
        # Set good agent to succeed via process (the method the scheduler calls)
        scheduler._agents["good_agent"].process.return_value = {
            "ticker": "AAPL",
            "agent_id": "good_agent",
            "fragment": "data",
        }
        # Make bad_agent.process raise
        scheduler._agents["bad_agent"].process.side_effect = Exception("Agent error")

        # Should not raise despite bad_agent failure
        result = await scheduler.run_single("AAPL")

        assert result is not None
        # Good agent should have been called
        scheduler._agents["good_agent"].process.assert_called()


class TestSweepStats:
    """Test sweep statistics."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        agents = {
            "agent_1": AsyncMock(),
            "agent_2": AsyncMock(),
        }
        fragment_store = MagicMock()
        fragment_store.append.return_value = True
        fragment_store.append_many.return_value = 1
        watchlist = Watchlist.default()

        return PerceptionScheduler(
            perception_agents=agents,
            fragment_store=fragment_store,
            watchlist=watchlist,
            inter_ticker_delay=0.01,
            inter_agent_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_sweep_stats_populated(self, scheduler):
        """Sweep stats are populated after sweep."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {"agent_id": "test", "fragment": "data"}

        result = await scheduler.run_full_sweep()

        assert "sweep_start" in result
        assert "sweep_end" in result
        assert "duration_seconds" in result
        assert "tickers_processed" in result
        assert "agents_run" in result
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_sweep_stats_stored_in_scheduler(self, scheduler):
        """Last sweep stats are stored in scheduler."""
        for agent in scheduler._agents.values():
            agent.process.return_value = {"agent_id": "test", "fragment": "data"}

        await scheduler.run_full_sweep()

        assert scheduler._last_sweep_time is not None
        assert len(scheduler._last_sweep_stats) > 0
