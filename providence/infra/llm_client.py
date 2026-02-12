"""LLM client for Adaptive agents.

Provides async Anthropic API client for Research Agents that use LLMs.
Handles retry logic, rate limiting, timeout, and structured JSON output.

Spec Reference: Technical Spec v2.3, Section 4.2
"""

import asyncio
import json
import os
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog

from providence.exceptions import ExternalAPIError

logger = structlog.get_logger()

# Default config
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the LLM client interface.

    All LLM clients (Anthropic, OpenAI, etc.) must implement this
    interface so agents can accept any provider interchangeably.
    """

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a prompt and return parsed JSON response."""
        ...


class AnthropicClient:
    """Async client for Anthropic's Messages API.

    Used by Adaptive agents (COGNIT-FUNDAMENTAL, COGNIT-MACRO, etc.)
    to send structured prompts and receive JSON responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._base_url = "https://api.anthropic.com/v1"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a prompt to Claude and return parsed JSON response.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: User message with context and query.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            ExternalAPIError: If API call fails after all retries.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/messages",
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 429:
                    # Rate limited — retry with backoff
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited by Anthropic API",
                        attempt=attempt,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code >= 500:
                    # Server error — retry
                    logger.warning(
                        "Anthropic API server error",
                        status_code=response.status_code,
                        attempt=attempt,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"Anthropic API error: {response.status_code}",
                        service="anthropic",
                        status_code=response.status_code,
                    )

                # Parse the response
                data = response.json()
                content_text = self._extract_text(data)
                return self._parse_json_response(content_text)

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "Anthropic API timeout",
                    attempt=attempt,
                    timeout=self._timeout,
                )
            except ExternalAPIError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    "Anthropic API request failed",
                    attempt=attempt,
                    error=str(e),
                )

        raise ExternalAPIError(
            message=f"Anthropic API failed after {self._max_retries} attempts: {last_error}",
            service="anthropic",
            status_code=0,
        )

    def _extract_text(self, response_data: dict[str, Any]) -> str:
        """Extract text content from Anthropic Messages API response.

        Handles empty content arrays and refusal/error responses.
        """
        # Check for stop reason indicating refusal
        stop_reason = response_data.get("stop_reason")
        if stop_reason == "end_turn" and not response_data.get("content"):
            raise ExternalAPIError(
                message="Anthropic API returned empty content (possible refusal)",
                service="anthropic",
                status_code=200,
            )

        content = response_data.get("content", [])
        if not content:
            raise ExternalAPIError(
                message="Anthropic API returned empty content array",
                service="anthropic",
                status_code=200,
            )

        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        result = "\n".join(text_parts)
        if not result.strip():
            raise ExternalAPIError(
                message="Anthropic API returned no text content in response blocks",
                service="anthropic",
                status_code=200,
            )

        return result

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from model response text.

        Handles cases where the JSON is wrapped in markdown code blocks.
        """
        cleaned = text.strip()

        # Strip markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from LLM response",
                error=str(e),
                response_preview=cleaned[:200],
            )
            raise ExternalAPIError(
                message=f"Failed to parse LLM JSON response: {e}",
                service="anthropic",
                status_code=200,
            )
