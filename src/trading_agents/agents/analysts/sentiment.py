from trading_agents.agents.analysts.base import AnalystAgent


class SentimentAnalystAgent(AnalystAgent):
    role = "sentiment_analyst"
    analyst_type = "sentiment"
