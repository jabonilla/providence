"""PERCEPT-NEWS: News sentiment ingestion agent.

Ingests news sentiment data from Polygon.io news API and produces MarketStateFragments
with data_type=SENTIMENT_NEWS.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw news data from Polygon.io REST API
  2. VALIDATE    → Check schema completeness and data availability
  3. NORMALIZE   → Convert to NewsPayload
  4. VERSION     → Compute content hash, assign fragment_id
  5. STORE       → Return MarketStateFragment (Kafka is future work)
  6. ALERT       → If validation fails, set QUARANTINED and log
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.polygon_client import PolygonClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.utils.redaction import redact_error_message
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import NewsPayload

logger = structlog.get_logger()


class PerceptNews(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-NEWS agent — ingests news sentiment data.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, polygon_client: PolygonClient) -> None:
        super().__init__(
            agent_id="PERCEPT-NEWS",
            agent_type="perception",
            version="1.0.0",
        )
        self._polygon = polygon_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process news data for tickers specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch news for
            - limit: int — max articles per ticker (default 10)

        Returns:
            List of MarketStateFragments, one per ticker.
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        limit: int = context.metadata.get("limit", 10)

        if not tickers:
            raise AgentProcessingError(
                message="No tickers specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []
        for ticker in tickers:
            try:
                fragment = await self._process_ticker(ticker, limit)
                fragments.append(fragment)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process ticker",
                    agent_id=self.agent_id,
                    ticker=ticker,
                    error=str(e),
                )
                # Produce a quarantined fragment so the failure is tracked
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(ticker, safe_error)
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_ticker(self, ticker: str, limit: int) -> MarketStateFragment:
        """Run the full Perception loop for a single ticker.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch(ticker, limit)

        # Step 2: VALIDATE
        validation_status = self._validate(raw_data, ticker)

        # Step 3: NORMALIZE
        payload = self._normalize(raw_data)

        # Step 4: VERSION — content hash computed by MarketStateFragment
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment (STORE is future Kafka work)
        source_ts = self._extract_source_timestamp(raw_data)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=ticker,
            data_type=DataType.SENTIMENT_NEWS,
            schema_version="1.0.0",
            source_hash=source_hash,
            validation_status=validation_status,
            payload=payload,
        )

        logger.info(
            "Fragment produced",
            agent_id=self.agent_id,
            ticker=ticker,
            validation_status=validation_status.value,
            fragment_id=str(fragment.fragment_id),
            article_count=payload.get("article_count", 0),
        )

        return fragment

    async def _fetch(self, ticker: str, limit: int) -> dict[str, Any]:
        """Step 1: FETCH — Pull raw news data from Polygon.io."""
        try:
            return await self._polygon.get_ticker_news(ticker, limit)
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch {ticker} news from Polygon: {e}",
                agent_id=self.agent_id,
            ) from e

    def _validate(self, raw_data: dict[str, Any], ticker: str) -> ValidationStatus:
        """Step 2: VALIDATE — Check schema completeness and data availability.

        Returns VALID, PARTIAL, or QUARANTINED based on data quality.
        """
        results = raw_data.get("results")

        # No results at all → quarantine
        if not results or not isinstance(results, list) or len(results) == 0:
            logger.warning(
                "No results in API response — quarantining",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return ValidationStatus.QUARANTINED

        # Check if articles have sentiment data (insights field)
        articles_with_insights = 0
        for article in results:
            if isinstance(article, dict) and article.get("insights"):
                articles_with_insights += 1

        if articles_with_insights == 0:
            logger.warning(
                "No articles have sentiment insights — marking PARTIAL",
                agent_id=self.agent_id,
                ticker=ticker,
                total_articles=len(results),
            )
            return ValidationStatus.PARTIAL

        if articles_with_insights < len(results):
            logger.warning(
                "Some articles lack sentiment data — marking PARTIAL",
                agent_id=self.agent_id,
                ticker=ticker,
                articles_with_insights=articles_with_insights,
                total_articles=len(results),
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert articles to NewsPayload format.

        Extracts relevant fields from Polygon news API response and
        builds a list of normalized article payloads.
        """
        results = raw_data.get("results", [])
        normalized_articles = []
        sentiment_scores = []

        for article in results:
            if not isinstance(article, dict):
                continue

            try:
                # Extract basic article fields
                source = article.get("publisher", {})
                if isinstance(source, dict):
                    source_name = source.get("name", "unknown")
                else:
                    source_name = "unknown"

                headline = article.get("title", "")
                summary = article.get("description", "")
                if summary:
                    summary = summary[:500]  # Truncate to 500 chars

                published_utc = article.get("published_utc", "")
                article_url = article.get("article_url")
                tickers_mentioned = article.get("tickers", [])
                keywords = article.get("keywords", [])

                # Extract sentiment and relevance from insights
                sentiment_score = 0.0
                relevance_score = 0.5
                insights = article.get("insights", [])

                if insights and isinstance(insights, list):
                    # Find insight matching the primary ticker (first one typically)
                    for insight in insights:
                        if isinstance(insight, dict):
                            # Try to use the first insight's sentiment
                            sentiment_str = insight.get("sentiment", "neutral").lower()
                            if sentiment_str == "positive":
                                sentiment_score = 0.5
                            elif sentiment_str == "negative":
                                sentiment_score = -0.5
                            elif sentiment_str == "neutral":
                                sentiment_score = 0.0
                            else:
                                # Try to parse as float
                                try:
                                    sentiment_score = float(sentiment_str)
                                    # Clamp to [-1.0, 1.0]
                                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                                except (ValueError, TypeError):
                                    sentiment_score = 0.0

                            # Extract relevance score
                            relevance_str = insight.get("relevance", "0.5")
                            try:
                                relevance_score = float(relevance_str)
                                relevance_score = max(0.0, min(1.0, relevance_score))
                            except (ValueError, TypeError):
                                relevance_score = 0.5

                            break  # Use first insight

                sentiment_scores.append(sentiment_score)

                # Create NewsPayload and convert to dict
                news_payload = NewsPayload(
                    source=source_name,
                    headline=headline,
                    summary=summary,
                    sentiment_score=sentiment_score,
                    relevance_score=relevance_score,
                    tickers_mentioned=tickers_mentioned,
                    published_utc=published_utc,
                    article_url=article_url,
                    keywords=keywords,
                )
                normalized_articles.append(news_payload.model_dump())

            except Exception as e:
                logger.warning(
                    "Failed to normalize article",
                    agent_id=self.agent_id,
                    error=str(e),
                )
                # Skip this article and continue
                continue

        # Compute average sentiment
        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores)
            if sentiment_scores
            else 0.0
        )

        return {
            "articles": normalized_articles,
            "article_count": len(normalized_articles),
            "avg_sentiment": avg_sentiment,
        }

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the raw API response for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _extract_source_timestamp(self, raw_data: dict[str, Any]) -> datetime:
        """Extract the source timestamp from the API response.

        Uses the published_utc of the most recent article if available.
        Falls back to current time if not available.
        """
        results = raw_data.get("results", [])
        if results and isinstance(results[0], dict):
            published_utc = results[0].get("published_utc")
            if published_utc:
                try:
                    # Parse ISO 8601 timestamp
                    return datetime.fromisoformat(published_utc.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

        return datetime.now(timezone.utc)

    def _create_quarantined_fragment(
        self, ticker: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=DataType.SENTIMENT_NEWS,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "articles": [], "article_count": 0, "avg_sentiment": 0.0},
        )

    def get_health(self) -> HealthStatus:
        """Report current health status."""
        if self._error_count_24h > 10:
            status = AgentStatus.UNHEALTHY
        elif self._error_count_24h > 3:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.HEALTHY

        return HealthStatus(
            agent_id=self.agent_id,
            status=status,
            last_run=self._last_run,
            last_success=self._last_success,
            error_count_24h=self._error_count_24h,
            avg_latency_ms=0.0,
        )
