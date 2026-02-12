"""Sample Polygon.io news API responses for testing PERCEPT-NEWS.

These fixtures replicate the structure of real Polygon.io news responses
so that tests can run without making actual API calls.
"""


def news_aapl() -> dict:
    """Sample Polygon.io news response for AAPL with multiple sentiments.

    Returns 3 articles with positive, neutral, and negative sentiments.
    """
    return {
        "status": "OK",
        "results": [
            {
                "id": "article_aapl_1",
                "author": "John Smith",
                "updated_utc": "2026-02-09T14:35:00Z",
                "published_utc": "2026-02-09T14:30:00Z",
                "title": "Apple Reports Record Q1 Revenue, Beats Analyst Expectations",
                "description": "Apple Inc. reported record first-quarter revenue of $119.6 billion, exceeding analyst expectations by 8%. The company attributed the strong performance to robust iPhone 15 sales and growing services revenue. CFO Luca Maestri indicated that gross margin expanded to 48.2%, reflecting operational efficiency gains.",
                "article_url": "https://reuters.com/tech/apple-q1-earnings-2026-02-09/",
                "publisher": {
                    "name": "Reuters",
                    "homepage_url": "https://reuters.com",
                    "logo_url": "https://reuters.com/logo.png",
                    "favicon_url": "https://reuters.com/favicon.ico",
                },
                "image_url": "https://reuters.com/apple-earnings.jpg",
                "tickers": ["AAPL"],
                "keywords": ["apple", "earnings", "revenue", "iphone", "record"],
                "insights": [
                    {
                        "ticker": "AAPL",
                        "sentiment": "positive",
                        "sentiment_reasoning": "Record revenue and margin expansion indicate strong market position",
                        "relevance": "0.95",
                    }
                ],
            },
            {
                "id": "article_aapl_2",
                "author": "Jane Doe",
                "updated_utc": "2026-02-08T10:15:00Z",
                "published_utc": "2026-02-08T10:00:00Z",
                "title": "Apple Faces Supply Chain Challenges in Asian Markets",
                "description": "Industry analysts report that Apple is encountering supply chain disruptions in Vietnam and Taiwan due to geopolitical tensions. The delays are expected to impact production of mid-range iPhone models in Q2 2026, potentially affecting revenue guidance.",
                "article_url": "https://bloomberg.com/tech/apple-supply-chain-2026-02-08/",
                "publisher": {
                    "name": "Bloomberg",
                    "homepage_url": "https://bloomberg.com",
                    "logo_url": "https://bloomberg.com/logo.png",
                    "favicon_url": "https://bloomberg.com/favicon.ico",
                },
                "image_url": "https://bloomberg.com/supply-chain.jpg",
                "tickers": ["AAPL"],
                "keywords": ["apple", "supply chain", "vietnam", "taiwan", "production"],
                "insights": [
                    {
                        "ticker": "AAPL",
                        "sentiment": "negative",
                        "sentiment_reasoning": "Supply chain disruptions present headwind to revenue",
                        "relevance": "0.85",
                    }
                ],
            },
            {
                "id": "article_aapl_3",
                "author": "Michael Chen",
                "updated_utc": "2026-02-07T16:45:00Z",
                "published_utc": "2026-02-07T16:30:00Z",
                "title": "Apple Maintains Market Share in Premium Smartphone Segment",
                "description": "Latest market research from IDC shows that Apple has maintained its 28% market share in the premium smartphone segment (devices over $800) during Q4 2025. The company's position remains stable despite increased competition from Samsung and OnePlus.",
                "article_url": "https://idc.com/research/apple-market-share-q4-2025/",
                "publisher": {
                    "name": "IDC Research",
                    "homepage_url": "https://idc.com",
                    "logo_url": "https://idc.com/logo.png",
                    "favicon_url": "https://idc.com/favicon.ico",
                },
                "image_url": "https://idc.com/market-research.jpg",
                "tickers": ["AAPL"],
                "keywords": ["apple", "market share", "smartphone", "premium"],
                "insights": [
                    {
                        "ticker": "AAPL",
                        "sentiment": "neutral",
                        "sentiment_reasoning": "Stable market share indicates steady competitive position",
                        "relevance": "0.72",
                    }
                ],
            },
        ],
        "count": 3,
        "next_url": "https://api.polygon.io/v2/reference/news?ticker=AAPL&limit=3&sort=published_utc&order=desc&cursor=...",
    }


def news_empty() -> dict:
    """Polygon response with no results (e.g., newly listed ticker with no news)."""
    return {
        "status": "OK",
        "results": [],
        "count": 0,
        "next_url": None,
    }


def news_no_insights() -> dict:
    """Polygon response with articles that lack sentiment insights (insights field missing).

    This represents PARTIAL validation status — articles exist but lack sentiment data.
    """
    return {
        "status": "OK",
        "results": [
            {
                "id": "article_partial_1",
                "author": "Sarah Johnson",
                "updated_utc": "2026-02-09T12:00:00Z",
                "published_utc": "2026-02-09T11:45:00Z",
                "title": "Tech Industry Hosts Annual Developer Conference",
                "description": "The annual tech industry conference brings together over 10,000 developers and executives to discuss emerging technologies and industry trends. Major companies are expected to announce new products and services.",
                "article_url": "https://techconf.com/2026-conference/",
                "publisher": {
                    "name": "Tech Weekly",
                    "homepage_url": "https://techweekly.com",
                    "logo_url": "https://techweekly.com/logo.png",
                    "favicon_url": "https://techweekly.com/favicon.ico",
                },
                "image_url": "https://techweekly.com/conference.jpg",
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "keywords": ["conference", "technology", "innovation"],
                # Note: NO insights field — this is the key difference
            },
            {
                "id": "article_partial_2",
                "author": "Robert Williams",
                "updated_utc": "2026-02-08T09:30:00Z",
                "published_utc": "2026-02-08T09:15:00Z",
                "title": "Smartphone Market Growth Slows in Developed Markets",
                "description": "Market analysts report that smartphone sales growth has slowed in North America and Europe but remains robust in emerging markets. Consumer upgrade cycles are extending as devices last longer.",
                "article_url": "https://marketresearch.com/smartphone-trends-2026/",
                "publisher": {
                    "name": "Market Research Daily",
                    "homepage_url": "https://marketresearch.com",
                    "logo_url": "https://marketresearch.com/logo.png",
                    "favicon_url": "https://marketresearch.com/favicon.ico",
                },
                "image_url": "https://marketresearch.com/market-trends.jpg",
                "tickers": ["AAPL", "SAMSUNG"],
                "keywords": ["smartphone", "market", "growth", "sales"],
                # Note: NO insights field
            },
        ],
        "count": 2,
        "next_url": None,
    }


def news_multi_ticker() -> dict:
    """Polygon response with articles mentioning multiple tickers.

    Articles can reference multiple companies and have multiple insights.
    """
    return {
        "status": "OK",
        "results": [
            {
                "id": "article_multi_1",
                "author": "Lisa Anderson",
                "updated_utc": "2026-02-09T13:20:00Z",
                "published_utc": "2026-02-09T13:00:00Z",
                "title": "Apple and Microsoft Announce Strategic Cloud Partnership",
                "description": "In a landmark announcement, Apple and Microsoft revealed a multi-year strategic partnership to integrate Microsoft's cloud services with Apple's ecosystem. The deal is expected to drive revenue growth for both companies and enhance enterprise customer offerings.",
                "article_url": "https://reuters.com/tech/apple-microsoft-partnership-2026-02-09/",
                "publisher": {
                    "name": "Reuters",
                    "homepage_url": "https://reuters.com",
                    "logo_url": "https://reuters.com/logo.png",
                    "favicon_url": "https://reuters.com/favicon.ico",
                },
                "image_url": "https://reuters.com/partnership.jpg",
                "tickers": ["AAPL", "MSFT"],
                "keywords": ["partnership", "cloud", "enterprise", "integration"],
                "insights": [
                    {
                        "ticker": "AAPL",
                        "sentiment": "positive",
                        "sentiment_reasoning": "Strategic partnership expands ecosystem and revenue streams",
                        "relevance": "0.88",
                    },
                    {
                        "ticker": "MSFT",
                        "sentiment": "positive",
                        "sentiment_reasoning": "Partnership enhances cloud competitiveness against AWS",
                        "relevance": "0.82",
                    },
                ],
            },
            {
                "id": "article_multi_2",
                "author": "David Martinez",
                "updated_utc": "2026-02-07T11:10:00Z",
                "published_utc": "2026-02-07T11:00:00Z",
                "title": "Intel Reports Strong Earnings; AMD Gains Market Share in Data Centers",
                "description": "Intel reported Q4 earnings that met analyst expectations, but data center revenue growth lagged industry averages. Meanwhile, AMD continues to gain market share in high-performance compute segments, particularly in AI and machine learning applications.",
                "article_url": "https://bloomberg.com/tech/intel-amd-earnings-2026-02-07/",
                "publisher": {
                    "name": "Bloomberg",
                    "homepage_url": "https://bloomberg.com",
                    "logo_url": "https://bloomberg.com/logo.png",
                    "favicon_url": "https://bloomberg.com/favicon.ico",
                },
                "image_url": "https://bloomberg.com/semiconductor.jpg",
                "tickers": ["INTC", "AMD"],
                "keywords": ["earnings", "data center", "market share", "semiconductor"],
                "insights": [
                    {
                        "ticker": "INTC",
                        "sentiment": "negative",
                        "sentiment_reasoning": "Lagging data center growth threatens core business margin",
                        "relevance": "0.91",
                    },
                    {
                        "ticker": "AMD",
                        "sentiment": "positive",
                        "sentiment_reasoning": "Market share gains in high-growth AI segment",
                        "relevance": "0.93",
                    },
                ],
            },
        ],
        "count": 2,
        "next_url": None,
    }
