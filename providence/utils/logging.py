"""Structured logging setup for Providence.

Uses structlog for JSON-structured logging with correlation IDs,
agent IDs, and timestamps in every log entry.
"""

import structlog


def configure_logging(json_output: bool = True) -> None:
    """Configure structlog for the Providence system.

    Args:
        json_output: If True (default), render logs as JSON.
                     If False, use console-friendly output for development.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(agent_id: str, correlation_id: str | None = None) -> structlog.BoundLogger:
    """Get a logger bound with agent_id and optional correlation_id.

    Args:
        agent_id: ID of the agent requesting the logger.
        correlation_id: Optional correlation ID for tracing across agents.

    Returns:
        A structlog BoundLogger with agent_id and correlation_id bound.
    """
    logger = structlog.get_logger()
    logger = logger.bind(agent_id=agent_id)
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
    return logger
