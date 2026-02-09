"""Providence exception hierarchy.

All custom exceptions inherit from ProvidenceError, allowing callers
to catch broad or specific error categories as needed.
"""


class ProvidenceError(Exception):
    """Base exception for all Providence errors."""

    def __init__(self, message: str = "", agent_id: str | None = None) -> None:
        self.agent_id = agent_id
        super().__init__(message)


class SchemaValidationError(ProvidenceError):
    """Raised when data fails schema validation.

    Examples: invalid MarketStateFragment, malformed BeliefObject,
    missing required fields.
    """


class AgentProcessingError(ProvidenceError):
    """Raised when an agent fails during processing.

    Examples: LLM call failure, computation error, timeout.
    """


class DataIngestionError(ProvidenceError):
    """Raised when data ingestion from an external source fails.

    Examples: Polygon.io API error, SEC EDGAR fetch failure,
    malformed response data.
    """


class ExternalAPIError(ProvidenceError):
    """Raised when an external API call fails.

    Examples: HTTP timeout, rate limiting, authentication failure,
    unexpected response format.
    """

    def __init__(
        self,
        message: str = "",
        agent_id: str | None = None,
        status_code: int | None = None,
        service: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.service = service
        super().__init__(message, agent_id)


class ContextAssemblyError(ProvidenceError):
    """Raised when CONTEXT-SVC fails to assemble agent context.

    Examples: no fragments available, token budget exceeded,
    invalid agent configuration.
    """


class ConstraintViolationError(ProvidenceError):
    """Raised when a portfolio or execution constraint is violated.

    Examples: position limit exceeded, sector concentration breach,
    gross exposure over limit, drawdown proximity.
    """

    def __init__(
        self,
        message: str = "",
        agent_id: str | None = None,
        constraint: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
    ) -> None:
        self.constraint = constraint
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(message, agent_id)
