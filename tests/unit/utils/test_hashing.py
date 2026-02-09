"""Tests for hashing utilities.

Validates deterministic hashing, context window hash computation,
and edge cases.
"""

from datetime import datetime, timezone

from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.utils.hashing import compute_content_hash, compute_context_window_hash


# ===================================================================
# compute_content_hash Tests
# ===================================================================
class TestComputeContentHash:
    """Tests for compute_content_hash utility."""

    def test_dict_produces_valid_hash(self) -> None:
        """Dict input should produce a 64-char hex SHA-256."""
        result = compute_content_hash({"ticker": "AAPL", "price": 186.90})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        """Same input should always produce same hash."""
        data = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        assert compute_content_hash(data) == compute_content_hash(data)

    def test_key_order_invariant(self) -> None:
        """Key order should not affect the hash."""
        h1 = compute_content_hash({"z": 26, "a": 1, "m": 13})
        h2 = compute_content_hash({"a": 1, "m": 13, "z": 26})
        assert h1 == h2

    def test_different_data_different_hash(self) -> None:
        """Different data should produce different hashes."""
        h1 = compute_content_hash({"value": 100})
        h2 = compute_content_hash({"value": 200})
        assert h1 != h2

    def test_empty_dict(self) -> None:
        """Empty dict should produce a valid hash."""
        result = compute_content_hash({})
        assert len(result) == 64

    def test_list_input(self) -> None:
        """List input should produce a valid hash."""
        result = compute_content_hash([1, 2, 3])
        assert len(result) == 64

    def test_string_input(self) -> None:
        """String input should produce a valid hash."""
        result = compute_content_hash("hello")
        assert len(result) == 64

    def test_pydantic_model_input(self, sample_fragment: MarketStateFragment) -> None:
        """Pydantic model input should use model_dump for hashing."""
        result = compute_content_hash(sample_fragment)
        assert len(result) == 64
        # Same fragment should produce same hash
        assert result == compute_content_hash(sample_fragment)


# ===================================================================
# compute_context_window_hash Tests
# ===================================================================
class TestComputeContextWindowHash:
    """Tests for compute_context_window_hash utility."""

    def _make_fragment(self, payload: dict) -> MarketStateFragment:
        """Helper to create a fragment with a given payload."""
        return MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
            data_type=DataType.PRICE_OHLCV,
            payload=payload,
        )

    def test_valid_hash_from_fragments(self) -> None:
        """Should produce a valid 64-char hex hash."""
        f1 = self._make_fragment({"close": 185.0})
        f2 = self._make_fragment({"close": 186.0})
        result = compute_context_window_hash([f1, f2])
        assert len(result) == 64

    def test_deterministic(self) -> None:
        """Same fragments should produce same hash."""
        f1 = self._make_fragment({"close": 185.0})
        f2 = self._make_fragment({"close": 186.0})
        h1 = compute_context_window_hash([f1, f2])
        h2 = compute_context_window_hash([f1, f2])
        assert h1 == h2

    def test_order_invariant(self) -> None:
        """Fragment order should not affect the hash (sorted internally)."""
        f1 = self._make_fragment({"close": 185.0})
        f2 = self._make_fragment({"close": 186.0})
        h_forward = compute_context_window_hash([f1, f2])
        h_reverse = compute_context_window_hash([f2, f1])
        assert h_forward == h_reverse

    def test_different_fragments_different_hash(self) -> None:
        """Different fragment sets should produce different hashes."""
        f1 = self._make_fragment({"close": 185.0})
        f2 = self._make_fragment({"close": 186.0})
        f3 = self._make_fragment({"close": 187.0})
        h12 = compute_context_window_hash([f1, f2])
        h13 = compute_context_window_hash([f1, f3])
        assert h12 != h13

    def test_empty_list(self) -> None:
        """Empty fragment list should produce a valid sentinel hash."""
        result = compute_context_window_hash([])
        assert len(result) == 64

    def test_single_fragment(self) -> None:
        """Single fragment should produce a valid hash."""
        f1 = self._make_fragment({"close": 185.0})
        result = compute_context_window_hash([f1])
        assert len(result) == 64

    def test_same_fragments_as_fixture(self, sample_fragment: MarketStateFragment) -> None:
        """Should work with fixture-provided fragments."""
        result = compute_context_window_hash([sample_fragment])
        assert len(result) == 64
        # Same fragment twice should be deterministic
        assert result == compute_context_window_hash([sample_fragment])
