# Trading Agents Framework

A multi-agent LLM trading framework, in the spirit of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents):
a Screener, four Analysts (fundamentals, sentiment, news, technical), a
Researcher, a Trader, a Risk Manager, and a Portfolio Manager collaborate
on each trading decision.

It has a fully testable Python core, a simulated (paper) broker with a
backtest runner, and adapters for Interactive Brokers (paper accounts by
default) and Zerodha Kite (orders disabled by default). A live trading
loop and a low-latency Rust/C++ execution service are future work — see
"Out of scope" below.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env   # then fill in ANTHROPIC_API_KEY, or set LLM_PROVIDER=ollama
```

## Running tests

```powershell
.venv\Scripts\pytest              # unit + backtest-smoke tests, no network, no API calls
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
```

Two live smoke tests are excluded by default (see `addopts` in
`pyproject.toml`). `test_live_llm_smoke.py` makes one real, billed call to
the Anthropic API (needs `ANTHROPIC_API_KEY` in `.env`);
`test_live_ollama_smoke.py` makes one call to a local Ollama server (needs
the model pulled):

```powershell
.venv\Scripts\pytest -m live_llm      # Claude
.venv\Scripts\pytest -m live_ollama   # local Ollama model
```

## Choosing the LLM

Agents run on Claude by default. To run them on a local model instead,
install [Ollama](https://ollama.com), pull a model, and set the provider
in `.env`:

```powershell
ollama pull llama3.1:8b
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b        # used for every role
```

No API key is needed with Ollama. `LLM_ROLE_OVERRIDES` still works and
takes Ollama model names, e.g. `{"researcher": "qwen3:32b"}` to give the
high-stakes roles a larger model. Before a run the CLI checks that the
server is reachable and every configured model is pulled.

Local models are less reliable at structured output than Claude, so the
Ollama client constrains generation to each agent's JSON schema, retries
once with the validation error when the output still doesn't fit, and
caps in-flight requests (`OLLAMA_MAX_CONCURRENCY`, default 2) so a local
server isn't flooded. Expect slower runs on CPU-only machines
(`OLLAMA_TIMEOUT_SECONDS`, default 300).

## Running a backtest

```powershell
.venv\Scripts\trading-agents run-backtest --universe AAPL,MSFT --start 2024-01-01 --end 2024-06-01
```

This fetches historical OHLCV via `yfinance` (cached as parquet under
`data/cache/`), runs the full agent pipeline day-by-day, and prints
summary stats (total return, Sharpe, max drawdown, win rate, trade count).
Every order is simulated through `PaperBrokerAdapter` — nothing here
touches a real brokerage account.

## Interactive Brokers

`InteractiveBrokersAdapter` (`brokers/ib.py`) talks to a running TWS or IB
Gateway through its socket API, via the optional `ib_async` dependency:

```powershell
.venv\Scripts\pip install -e ".[ib]"
```

In TWS, enable the API (Configure > API > Settings > "Enable ActiveX and
Socket Clients") and note the port: **paper TWS 7497, paper Gateway
4002** (live: 7496 / 4001). Then set it in `.env` (`IB_PORT`, and
`IB_ACCOUNT` if your login manages several accounts) and check the
connection — read-only, so it cannot place orders:

```powershell
.venv\Scripts\trading-agents ib-check
```

Safety and behavior:
- The adapter refuses non-paper accounts (paper IDs start with `D`)
  unless built with `allow_live=True`.
- Quantities are floored to whole shares; an order that rounds to zero is
  rejected rather than silently changed.
- `place_order` waits briefly for IB to accept or reject. A market order
  placed outside regular hours stays queued and comes back as
  `submitted`; poll `get_order_status` for the fill. Rejections raise
  `OrderRejectedError` with IB's reason.
- Only US stocks are mapped (`SMART` routing, USD).

Nothing in the agent pipeline places IB orders yet — the backtest always
uses `PaperBrokerAdapter`. Wiring a real broker into a trading loop, with
its own safeguards, is the next piece of future work.

## Zerodha Kite

`KiteBrokerAdapter` and `KiteHistoricalDataProvider` (`brokers/kite.py`)
call the Kite Connect v3 REST API directly (no extra dependency). You need
a Kite Connect app (API key and secret, with its redirect URL set) in
`.env` as `KITE_API_KEY` / `KITE_API_SECRET`.

**Kite has no paper trading: every order is real money.** The adapter
refuses to place orders unless it is built with `allow_live=True`; reads
always work. Nothing in the pipeline places Kite orders yet.

Daily login — Kite access tokens expire at 6 AM IST:

```powershell
.venv\Scripts\trading-agents kite-login                         # prints the login URL
.venv\Scripts\trading-agents kite-login --request-token <token> # from the redirect
# add the printed KITE_ACCESS_TOKEN=... line to .env, then (read-only):
.venv\Scripts\trading-agents kite-check
```

Behavior and requirements:
- Under SEBI's retail algo rules, Zerodha only accepts API **orders** from
  a static IP registered with them; reads work from anywhere. A refusal
  that mentions the IP says so.
- Market orders are sent with Kite's automatic market protection
  (`market_protection=-1`), which API market orders require.
- Tickers are NSE trading symbols (`INFY`, not yfinance's `INFY.NS`);
  quantities are floored to whole shares; the product is CNC (delivery).
- Positions combine settled and T1 holdings with today's net positions.
- Historical candles may need the historical-data add-on on your Kite
  Connect app; long ranges are fetched in chunks within Kite's per-request
  limits, and requests are throttled to Kite's published rate limits.
- Kite's order APIs don't report charges, so `Order.commission` is 0.

## Architecture

- `contracts/` — pydantic data contracts shared by every layer (market
  bars, analyst reports, trade proposals, orders, portfolio state,
  backtest results).
- `llm/` — the `LLMClient` abstraction with Claude (`client.py`) and
  Ollama (`ollama_client.py`) implementations, structured output parsing,
  per-role model config, and prompt templates under `llm/prompts/*.md`.
- `agents/` — one module per agent role. Each agent's LLM output is a
  narrow, role-specific pydantic model; deterministic fields (timestamps,
  echoed inputs, hard risk/cash limits) are filled in by Python, not the
  LLM.
- `brokers/` — the `BrokerAdapter` Protocol, `PaperBrokerAdapter`
  (simulated), `InteractiveBrokersAdapter` (`brokers/ib.py`), and
  `KiteBrokerAdapter` plus `KiteHistoricalDataProvider` (`brokers/kite.py`).
  A future low-latency execution service implements the same Protocol
  without any orchestration-layer changes.
- `data/` — `HistoricalDataProvider` Protocol, the `YFinanceProvider`
  implementation, and a parquet-backed cache. `data/evidence/` builds the
  fundamentals/sentiment/news analysts' evidence (see below).
- `orchestration/` — `DecisionCycleOrchestrator`, a hand-rolled async
  pipeline (Screener → 4 parallel Analysts per ticker → Researcher →
  Trader → RiskManager → PortfolioManager) over a shared state object.
- `backtest/` — `BacktestRunner` (drives the orchestrator day-by-day
  against `PaperBrokerAdapter`) and `metrics.py` (return, Sharpe, max
  drawdown, win rate).

## Analyst evidence

The fundamentals, sentiment, and news analysts get their evidence from
`DefaultEvidenceProvider` (`data/evidence/`). Every record carries the time
it became public, and only records public **strictly before** the decision
time are used. Backtest decisions use midnight of the trading day, so
nothing released that day (e.g. after-close earnings) can leak in.

| Analyst | Source | History usable in backtests |
|---|---|---|
| Fundamentals | yfinance earnings releases (EPS, surprise, TTM EPS, P/E) | ~20 years |
| | yfinance quarterly statements (margins, revenue growth, leverage) | last ~5 quarters only |
| Sentiment | yfinance analyst upgrades/downgrades and price targets (90-day window) | years |
| | short interest, consensus rating | live only (current snapshot) |
| News | SEC EDGAR 8-K material events and 10-Q/10-K filings (30-day window) | years |
| | Yahoo headlines (7-day window) | live only (last few days) |

SEC filings are fetched only when `SEC_USER_AGENT` is set in `.env`. SEC's
fair-access policy requires it to name you with a contact email, e.g.
`SEC_USER_AGENT=Jane Doe jane@example.com`. When a source is unavailable
or fails, the evidence says so explicitly (`note` / `source_errors`), and
the analyst prompts tell the model to return a low-confidence neutral
signal rather than reading meaning into missing data.

Raw data is cached in memory (6 hours) so a backtest fetches each source
once per ticker per run. yfinance values can be restated after the fact,
so fundamentals are close to, but not exactly, what was known at the time.

## Known gaps in this phase

- Backtests more than about a year back get no margin/leverage
  fundamentals (yfinance only serves the last ~5 quarterly statements) and
  no headlines; EPS, analyst actions, and SEC filings still apply.
- The backtest runner fills orders at the simulated day's close with fixed
  commission/slippage — there's no intraday or tick-level simulation.

## Out of scope for this phase (future work)

- A **Rust/C++ low-latency execution service** for live order routing,
  reached from Python over gRPC or a local socket/queue, implementing the
  same `BrokerAdapter` Protocol.
- Live trading of any kind — everything today runs against
  `PaperBrokerAdapter`.
- **Further LLM providers.** Claude and local Ollama models are
  supported; `LLMClient` is a `Protocol`, so another implementation can be
  added without touching `agents/` or `orchestration/`.
