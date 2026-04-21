"""Perception subsystem — frozen agents that ingest external market data."""

from providence.agents.perception.alphavantage import PerceptAlphaVantage
from providence.agents.perception.cds import PerceptCds
from providence.agents.perception.factors import PerceptFactors
from providence.agents.perception.filing import PerceptFiling
from providence.agents.perception.fundflow import PerceptFundFlow
from providence.agents.perception.macro import PerceptMacro
from providence.agents.perception.news import PerceptNews
from providence.agents.perception.options import PerceptOptions
from providence.agents.perception.price import PerceptPrice
from providence.agents.perception.yfinance_agent import PerceptYFinance

__all__ = [
    "PerceptAlphaVantage",
    "PerceptCds",
    "PerceptFactors",
    "PerceptFiling",
    "PerceptFundFlow",
    "PerceptMacro",
    "PerceptNews",
    "PerceptOptions",
    "PerceptPrice",
    "PerceptYFinance",
]
