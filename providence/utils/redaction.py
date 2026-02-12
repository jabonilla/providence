"""Redaction utilities for sensitive data in error messages and logs.

Strips API keys, tokens, and credentials from URLs and strings
before they are stored in fragment payloads or logged.

Spec Reference: Security best practice — prevent API key leakage
into QUARANTINED fragment payloads and structured logs.
"""

import re
from urllib.parse import parse_qs, urlparse, urlunparse


# URL parameters that should be redacted
SENSITIVE_PARAMS = {
    "apikey",
    "api_key",
    "apiKey",
    "key",
    "token",
    "secret",
    "password",
    "access_token",
    "authorization",
}

# Regex patterns for API keys in free text
_KEY_PATTERNS = [
    # Polygon-style: apiKey=XXXXX in URL
    re.compile(r"(apiKey=)[^\s&]+", re.IGNORECASE),
    # Generic key=value patterns
    re.compile(r"(api_key=)[^\s&]+", re.IGNORECASE),
    re.compile(r"(token=)[^\s&]+", re.IGNORECASE),
    re.compile(r"(secret=)[^\s&]+", re.IGNORECASE),
    # Bearer tokens
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
    # Anthropic API keys
    re.compile(r"(sk-ant-)[^\s\"']+"),
]


def redact_url(url: str) -> str:
    """Remove sensitive query parameters from a URL.

    Args:
        url: URL that may contain API keys in query parameters.

    Returns:
        URL with sensitive parameter values replaced by '[REDACTED]'.
    """
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        redacted = {}
        for key, values in params.items():
            if key.lower() in {p.lower() for p in SENSITIVE_PARAMS}:
                redacted[key] = ["[REDACTED]"]
            else:
                redacted[key] = values

        parts = []
        for key, values in redacted.items():
            for v in values:
                parts.append(f"{key}={v}")
        new_query = "&".join(parts)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        # If URL parsing fails, fall back to regex redaction
        return redact_error_message(url)


def redact_error_message(message: str) -> str:
    """Strip potential API keys and tokens from an error message.

    Args:
        message: Error string that may contain leaked credentials.

    Returns:
        Message with sensitive values replaced by '[REDACTED]'.
    """
    result = message
    for pattern in _KEY_PATTERNS:
        result = pattern.sub(r"\1[REDACTED]", result)
    return result
