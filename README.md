# Trading Agents Framework

A multi-agent LLM trading framework, in the spirit of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents):
a Screener, four Analysts (fundamentals, sentiment, news, technical), a
Researcher, a Trader, a Risk Manager, and a Portfolio Manager collaborate
on each trading decision.

**This is Phase 1**: a fully testable Python core with a simulated
(paper) broker and a backtest runner. Real broker connectivity
(Interactive Brokers, Zerodha Kite) and a low-latency Rust/C++ execution
service are future phases — see "Out of scope" below.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Running tests

```powershell
.venv\Scripts\pytest              # unit + backtest-smoke tests, no network, no API calls
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
```

`tests/integration/test_live_llm_smoke.py` is excluded by default (see
`addopts` in `pyproject.toml`). It makes one real, billed call to the
Anthropic API. Run it explicitly once you have a real `ANTHROPIC_API_KEY`
in `.env`:

```powershell
.venv\Scripts\pytest -m live_llm
```

## Running a backtest

```powershell
.venv\Scripts\trading-agents run-backtest --universe AAPL,MSFT --start 2024-01-01 --end 2024-06-01
```

This fetches historical OHLCV via `yfinance` (cached as parquet under
`data/cache/`), runs the full agent pipeline day-by-day, and prints
summary stats (total return, Sharpe, max drawdown, win rate, trade count).
Every order is simulated through `PaperBrokerAdapter` — nothing here
touches a real brokerage account.

## Architecture

- `contracts/` — pydantic data contracts shared by every layer (market
  bars, analyst reports, trade proposals, orders, portfolio state,
  backtest results).
- `llm/` — the `LLMClient` abstraction, structured (tool-use) output
  parsing, per-role model config, and prompt templates under
  `llm/prompts/*.md`.
- `agents/` — one module per agent role. Each agent's LLM output is a
  narrow, role-specific pydantic model; deterministic fields (timestamps,
  echoed inputs, hard risk/cash limits) are filled in by Python, not the
  LLM.
- `brokers/` — the `BrokerAdapter` Protocol and `PaperBrokerAdapter` (the
  only adapter implemented in this phase). Real IB/Kite adapters, and a
  future low-latency execution service, implement the same Protocol
  without any orchestration-layer changes.
- `data/` — `HistoricalDataProvider` Protocol, the `YFinanceProvider`
  implementation, and a parquet-backed cache.
- `orchestration/` — `DecisionCycleOrchestrator`, a hand-rolled async
  pipeline (Screener → 4 parallel Analysts per ticker → Researcher →
  Trader → RiskManager → PortfolioManager) over a shared state object.
- `backtest/` — `BacktestRunner` (drives the orchestrator day-by-day
  against `PaperBrokerAdapter`) and `metrics.py` (return, Sharpe, max
  drawdown, win rate).

## Known gaps in this phase

- **Fundamentals/sentiment/news data sourcing isn't wired up.** Those
  three analysts are fully implemented and tested (see
  `tests/unit/agents/test_analysts.py`), but `DecisionCycleOrchestrator`
  defaults their evidence to an empty dict unless a caller supplies
  `fundamentals_evidence`/`sentiment_evidence`/`news_evidence`. Only the
  Technical analyst has a real data path today (computed from `yfinance`
  OHLCV via `indicators/technical.py`).
- The backtest runner fills orders at the simulated day's close with fixed
  commission/slippage — there's no intraday or tick-level simulation.

## Out of scope for this phase (future work)

- **Interactive Brokers** adapter (`ib_async`) and **Zerodha Kite**
  adapter (`kiteconnect`) — `BrokerAdapter`'s lifecycle methods
  (`connect`/`disconnect`/`is_connected`) exist specifically so these can
  be added without touching `agents/` or `orchestration/`.
- A **Rust/C++ low-latency execution service** for live order routing,
  reached from Python over gRPC or a local socket/queue, implementing the
  same `BrokerAdapter` Protocol.
- Live trading of any kind — everything today runs against
  `PaperBrokerAdapter`.
- **Support for LLM providers other than Anthropic.** `LLMClient` is a
  `Protocol` (same shape as `BrokerAdapter`), so a second implementation
  (e.g. OpenAI, or a local model via Ollama) could be added without
  touching `agents/` or `orchestration/`. Phase 1 deliberately shipped
  Claude-only; this is tracked as future scope, not a regression.
