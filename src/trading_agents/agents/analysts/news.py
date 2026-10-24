from trading_agents.agents.analysts.base import AnalystAgent


class NewsAnalystAgent(AnalystAgent):
    role = "news_analyst"
    analyst_type = "news"
