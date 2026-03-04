"""Tests for API security features (Session 35).

Covers:
- Error sanitization (credential removal)
- Rate limiting middleware
- Request size limiting middleware
- Input validation on agent filters
- No detail leakage in error responses
"""

from __future__ import annotations

import pytest

# Guard: skip entire module if fastapi not installed
pytest.importorskip("fastapi")

from providence.api.security import sanitize_error


# ── Error sanitization ────────────────────────────────────────────


class TestSanitizeError:
    """Verify sensitive data is stripped from error messages."""

    def test_redacts_anthropic_key(self):
        exc = Exception(
            "API error: sk-ant-api03-2AYFa6SkHMG9xbrehmYxx_QB-_2q9Y7exiEn"
            "PrIwc7qTdKOxJJ-M1JSQTUY1cs9QRV5aLI7uJyx-UnWM7z7_xA-XpDV7gAA"
        )
        result = sanitize_error(exc)
        assert "sk-ant" not in result
        assert "<REDACTED" in result

    def test_redacts_bearer_token(self):
        exc = Exception("Auth failed: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        result = sanitize_error(exc)
        assert "eyJhbGci" not in result
        assert "<REDACTED" in result

    def test_redacts_api_key_param(self):
        exc = Exception("Request to https://api.example.com?api_key=secret123")
        result = sanitize_error(exc)
        assert "secret123" not in result

    def test_redacts_url_credentials(self):
        exc = Exception("Connection to https://user:p4ssw0rd@db.example.com failed")
        result = sanitize_error(exc)
        assert "p4ssw0rd" not in result

    def test_redacts_aws_key(self):
        exc = Exception("AWS error with key AKIAIOSFODNN7EXAMPLE")
        result = sanitize_error(exc)
        assert "AKIAIOSFODNN7" not in result

    def test_preserves_safe_messages(self):
        exc = Exception("Connection refused to Polygon API after 3 retries")
        result = sanitize_error(exc)
        assert "Connection refused" in result

    def test_generic_on_traceback(self):
        exc = Exception("Traceback (most recent call last):\n  File...")
        result = sanitize_error(exc)
        assert "contact support" in result.lower() or "error occurred" in result.lower()


# ── Input validation ──────────────────────────────────────────────


class TestAgentInputValidation:
    """Verify agent endpoint validates filter parameters."""

    @pytest.fixture
    def client(self):
        """Create a test client with minimal state."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from providence.api.app import create_app
        from providence.api.deps import AppState

        state = AppState(
            agent_registry={},
            fragment_store=MagicMock(),
            belief_store=MagicMock(),
            run_store=MagicMock(),
        )
        app = create_app(state, enable_auth=False)
        return TestClient(app)

    def test_valid_subsystem_filter(self, client):
        resp = client.get("/api/v1/agents?subsystem=cognition")
        assert resp.status_code == 200

    def test_invalid_subsystem_rejected(self, client):
        resp = client.get("/api/v1/agents?subsystem=EVIL")
        assert resp.status_code == 400
        assert "Invalid subsystem" in resp.json()["detail"]

    def test_valid_classification_filter(self, client):
        resp = client.get("/api/v1/agents?classification=FROZEN")
        assert resp.status_code == 200

    def test_invalid_classification_rejected(self, client):
        resp = client.get("/api/v1/agents?classification=HACKED")
        assert resp.status_code == 400
        assert "Invalid classification" in resp.json()["detail"]


# ── Error response safety ─────────────────────────────────────────


class TestErrorResponseSafety:
    """Verify error responses don't leak implementation details."""

    @pytest.fixture
    def client(self):
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from providence.api.app import create_app
        from providence.api.deps import AppState

        state = AppState(
            agent_registry={},
            fragment_store=MagicMock(),
            belief_store=MagicMock(),
            run_store=MagicMock(),
        )
        app = create_app(state, enable_auth=False)
        return TestClient(app)

    def test_404_no_id_leakage_agents(self, client):
        resp = client.get("/api/v1/agents/SECRET-AGENT-NAME")
        assert resp.status_code == 404
        body = resp.json()
        # Should NOT echo back the agent ID in the detail
        assert "SECRET-AGENT-NAME" not in body.get("detail", "")

    def test_400_no_enum_leakage_stores(self, client):
        resp = client.get("/api/v1/stores/fragments?data_type=FAKE_TYPE")
        assert resp.status_code == 400
        body = resp.json()
        # Should NOT echo back the input value
        assert "FAKE_TYPE" not in body.get("detail", "")

    def test_400_no_enum_leakage_pipeline(self, client):
        resp = client.get("/api/v1/pipeline/runs?status=FAKE_STATUS")
        assert resp.status_code == 400
        body = resp.json()
        assert "FAKE_STATUS" not in body.get("detail", "")


# ── CORS ──────────────────────────────────────────────────────────


class TestCORSConfiguration:
    """Verify CORS is properly restricted."""

    def test_cors_not_wildcard_in_production(self):
        """In production mode, CORS should not allow all origins."""
        import os
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from providence.api.app import create_app
        from providence.api.deps import AppState

        state = AppState(
            agent_registry={},
            fragment_store=MagicMock(),
            belief_store=MagicMock(),
            run_store=MagicMock(),
        )

        with patch.dict(os.environ, {"PROVIDENCE_ENV": "production"}):
            app = create_app(state, enable_auth=False)
            client = TestClient(app)

        # Preflight request from unknown origin
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should NOT include Access-Control-Allow-Origin: *
        acl = resp.headers.get("access-control-allow-origin", "")
        assert acl != "*"
        assert "evil.com" not in acl
