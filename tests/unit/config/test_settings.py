"""Tests for ProvidenceSettings — environment-based configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from providence.config.settings import ProvidenceSettings, get_settings


class TestDefaults:
    """Test default values when no env vars are set."""

    def test_default_log_level(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.log_level == "INFO"

    def test_default_data_dir(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.data_dir == Path("data")

    def test_default_interval(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.interval_seconds == 300

    def test_default_timeout(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.default_timeout == 120.0

    def test_default_skip_flags(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.skip_perception is False
        assert settings.skip_adaptive is False

    def test_default_feature_flags(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        assert settings.persist_storage is True
        assert settings.extract_beliefs is True


class TestEnvOverrides:
    """Test that environment variables override defaults."""

    def test_log_level_override(self):
        with patch.dict(os.environ, {"PROVIDENCE_LOG_LEVEL": "debug"}):
            settings = get_settings()
        assert settings.log_level == "DEBUG"

    def test_data_dir_override(self):
        with patch.dict(os.environ, {"PROVIDENCE_DATA_DIR": "/tmp/prov"}):
            settings = get_settings()
        assert settings.data_dir == Path("/tmp/prov")

    def test_interval_override(self):
        with patch.dict(os.environ, {"PROVIDENCE_INTERVAL_SECONDS": "60"}):
            settings = get_settings()
        assert settings.interval_seconds == 60

    def test_timeout_override(self):
        with patch.dict(os.environ, {"PROVIDENCE_DEFAULT_TIMEOUT": "30.5"}):
            settings = get_settings()
        assert settings.default_timeout == 30.5

    def test_skip_perception_true(self):
        with patch.dict(os.environ, {"PROVIDENCE_SKIP_PERCEPTION": "true"}):
            settings = get_settings()
        assert settings.skip_perception is True

    def test_skip_adaptive_yes(self):
        with patch.dict(os.environ, {"PROVIDENCE_SKIP_ADAPTIVE": "yes"}):
            settings = get_settings()
        assert settings.skip_adaptive is True

    def test_skip_flag_zero(self):
        with patch.dict(os.environ, {"PROVIDENCE_SKIP_PERCEPTION": "0"}):
            settings = get_settings()
        assert settings.skip_perception is False

    def test_persist_storage_false(self):
        with patch.dict(os.environ, {"PROVIDENCE_PERSIST_STORAGE": "false"}):
            settings = get_settings()
        assert settings.persist_storage is False


class TestApiKeyDetection:
    """Test API key availability checks."""

    def test_no_keys_available(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("POLYGON_API_KEY", "EDGAR_USER_AGENT", "FRED_API_KEY", "ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings()
            summary = settings.available_api_summary()
        assert summary["polygon"] is False
        assert summary["edgar"] is False
        assert summary["fred"] is False
        assert summary["anthropic"] is False

    def test_polygon_key_present(self):
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test-key"}):
            settings = get_settings()
            assert settings.has_polygon_key() is True

    def test_anthropic_key_present(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            settings = get_settings()
            assert settings.has_anthropic_key() is True

    def test_all_keys_present(self):
        env = {
            "POLYGON_API_KEY": "pk",
            "EDGAR_USER_AGENT": "Test/1.0 test@test.com",
            "FRED_API_KEY": "fk",
            "ANTHROPIC_API_KEY": "sk",
            "ALPACA_API_KEY": "ak",
            "ALPACA_SECRET_KEY": "ask",
        }
        with patch.dict(os.environ, env):
            settings = get_settings()
            summary = settings.available_api_summary()
            assert all(summary.values())


class TestImmutability:
    """Settings should be frozen (immutable)."""

    def test_cannot_modify_settings(self):
        settings = get_settings()
        with pytest.raises(Exception):
            settings.log_level = "DEBUG"
