# Trading Agents Framework

A multi-agent LLM trading framework, in the spirit of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents):
a Screener, four Analysts (fundamentals, sentiment, news, technical), a
Researcher, a Trader, a Risk Manager, and a Portfolio Manager collaborate
on each trading decision.

It has a fully testable Python core, a simulated (paper) broker with a
backtest runner, adapters for Interactive Brokers and Zerodha Kite, a
guarded live trading loop that runs in dry-run mode unless told
otherwise, and an optional Rust execution service (`execd`) that can sit
between the loop and the broker.

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
- The IB live loop takes its price bars from IB itself, so decisions use
  the prices IB fills at. That needs market-data permissions on the
  account; a paper account must be set (in Account Management) to share
  the live account's subscriptions. Without them IB returns no bars and the
  cycle reports the missing permissions; `IB_MARKET_DATA=yfinance` switches
  back to Yahoo prices (about 15 minutes delayed during the session).
  Fundamentals, sentiment, and news evidence still come from yfinance and
  SEC EDGAR either way.

The backtest always uses `PaperBrokerAdapter`; IB orders are placed only
by the live trading loop (below).

## Zerodha Kite

`KiteBrokerAdapter` and `KiteHistoricalDataProvider` (`brokers/kite.py`)
call the Kite Connect v3 REST API directly (no extra dependency). You need
a Kite Connect app (API key and secret, with its redirect URL set) in
`.env` as `KITE_API_KEY` / `KITE_API_SECRET`.

**Kite has no paper trading: every order is real money.** The adapter
refuses to place orders unless it is built with `allow_live=True`; reads
always work. Only the live trading loop in `--mode live` builds it that
way.

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

## Live trading loop

`trading-agents trade` runs one decision cycle per trading day at a
scheduled time — by default 30 minutes before the close (15:30 New York
for IB, 15:00 Mumbai for Kite; `LIVE_DECISION_TIME=HH:MM` to change) —
against the broker in `LIVE_BROKER` (`ib` or `kite`):

```powershell
.venv\Scripts\trading-agents trade --once            # one dry-run cycle now
.venv\Scripts\trading-agents trade                   # dry-run, every weekday
.venv\Scripts\trading-agents trade --mode paper      # IB paper account
.venv\Scripts\trading-agents trade --mode live       # REAL orders (see below)
```

| Mode | What happens | IB | Kite |
|---|---|---|---|
| `dry-run` (default) | Full cycle; orders it *would* send are journaled, nothing is sent | read-only connection | orders disabled |
| `paper` | Orders go to a paper account | paper accounts only | refused (no paper trading) |
| `live` | Real orders | needs both opt-ins | needs both opt-ins |

**Real money needs two independent opt-ins:** `--mode live` on the command
line *and* `ALLOW_LIVE_TRADING=true` in `.env`.

Guardrails, enforced in Python after the agents decide (they can only
shrink or block an order) and configurable in `.env`:

| Guardrail | Default | Setting |
|---|---|---|
| Max size of one buy | 5% of equity (whole shares) | `LIVE_MAX_ORDER_PCT` |
| Max orders sent per day | 10 | `LIVE_MAX_ORDERS_PER_DAY` |
| Stop trading for the day after a drawdown from the equity at the end of the previous cycle | 2% | `LIVE_MAX_DAILY_DRAWDOWN_PCT` |
| Kill switch: while this file exists, nothing is sent | `KILL_SWITCH` | `LIVE_KILL_SWITCH_PATH` |

These come on top of the agents' own limits (the risk manager's 10%
per-ticker position cap and the portfolio manager's cash check). Sells
are not size-capped, since they reduce exposure. The kill switch is
checked before every order, so creating the file mid-cycle stops the
orders not yet sent. The counters and the drawdown baseline are saved
under `data/live/` per broker, mode, and account, so restarting the loop
doesn't reset them and a dry or paper run never affects a live one.

Every cycle appends one JSON line to `data/live/<broker>-<mode>-journal.jsonl`:
each decision, any guardrail that trimmed or blocked it, and the order's
outcome. If a send fails in a way that leaves it unclear whether the broker
took the order (a timeout, a dropped connection), that order is journaled as
`unknown` and nothing more is sent that cycle; check the broker before
re-running. An order the broker accepted is always journaled with its order
id, even if the follow-up status check failed.

Days with no bar dated today (holidays) are skipped, as are tickers whose
data is stale, and so is a cycle that starts more than 15 minutes late
(e.g. the machine slept through the decision time), so it never trades on
a stale decision. An IB connection or a Kite token that has expired skips
that day's cycle; for Kite, run `kite-login` each morning. A restarted
`execd` is detected and reconnected (and re-checked) on the next cycle.

Known gaps: Kite symbols run on technical analysis only (the evidence
sources are US-centric), and there is no market-holiday calendar beyond
the "no bar today" check.

## Rust execution service (`execd`)

`execution_service/` is a Rust daemon that exposes the broker operations
(balance, positions, orders, history) over newline-delimited JSON on a
loopback TCP port. The live loop reaches it through
`ExecutionServiceAdapter` with `LIVE_BROKER=execd`. Backends:

- `paper`: a simulated ledger with immediate fills (whole shares,
  configurable slippage and commission).
- `kite`: Zerodha Kite Connect over one persistent HTTPS connection pool,
  with the same rules as the Python Kite adapter (orders refused unless
  started with `--allow-live`, `market_protection=-1`, whole shares,
  static-IP hint, rate limits, chunked history).

```powershell
cd execution_service
cargo build --release
$env:EXECD_TOKEN = "<at least 16 random characters>"
.\target\release\execd --backend paper              # simulated
.\target\release\execd --backend kite               # Kite reads only
.\target\release\execd --backend kite --allow-live  # Kite real orders
```

`--backend kite` reads `KITE_API_KEY` / `KITE_ACCESS_TOKEN` from the
environment and checks the session before it starts listening. Then in
`.env`: `LIVE_BROKER=execd`, the same `EXECD_TOKEN`, and
`EXECD_MARKET=nse` (Kite symbols) or `us`.

Safety:
- It only listens on loopback, and every request must carry the token,
  compared in constant time.
- Real orders need the daemon's `--allow-live` **and** the loop's two
  opt-ins.
- The Python adapter checks the daemon rather than trusting it: paper mode
  refuses unless the daemon runs `paper`, and live mode refuses unless it
  runs `kite` with orders enabled. Dry runs block orders before they leave
  Python.

On latency: at the default once-a-day cadence, the LLM calls take seconds
and orders are a handful per day, so `execd` makes no measurable speed
difference. What it adds is an order path isolated from the Python
process, a persistent broker connection, and an independent safety gate.
It is also the seam for any future higher-frequency strategy.

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
- `live/` — `LiveTradingLoop` (scheduling, modes, journal) and
  `guardrails.py` (order limits, drawdown halt, kill switch, persisted
  daily state); wired up by `cli/trade.py`.
- `execution_service/` — the Rust `execd` daemon (see below);
  `brokers/execution_service.py` is the Python adapter that talks to it.
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
once per ticker per run; a failing source is remembered for 5 minutes
rather than retried for every ticker on every day. yfinance values can be
restated after the fact, so fundamentals are close to, but not exactly,
what was known at the time.

Details that keep the evidence honest:
- Analyst price targets are stored by Yahoo as issued, while prices are
  split-adjusted; targets are rescaled with the split history so "implied
  upside" compares like with like (NVDA before its 2024 10-for-1 split
  otherwise shows about +1,000%).
- A quarter with no known earnings date is treated as public 90 days after
  it ends, the latest SEC deadline (10-K), so it can only be late, never
  early.
- Current-only values (short interest, consensus rating) are used only for
  a decision made right now, never for a past or recent backtest day.

## Known gaps in this phase

- Backtests more than about a year back get no margin/leverage
  fundamentals (yfinance only serves the last ~5 quarterly statements) and
  no headlines; EPS, analyst actions, and SEC filings still apply.
- The backtest runner fills orders at the simulated day's close with fixed
  commission/slippage — there's no intraday or tick-level simulation.

## Out of scope for this phase (future work)

- **IB in the Rust execution service.** `execd` fronts Kite (and a paper
  ledger); IB stays on the Python adapter, since reimplementing TWS's
  versioned binary socket protocol in Rust can't be verified without a
  live TWS.
- **Further LLM providers.** Claude and local Ollama models are
  supported; `LLMClient` is a `Protocol`, so another implementation can be
  added without touching `agents/` or `orchestration/`.
