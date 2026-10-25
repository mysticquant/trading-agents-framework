# execd

Local order-execution service for trading-agents. See the "Rust execution
service" section of the top-level README for setup and safety rules.

## Protocol

One JSON object per line in each direction, over a loopback TCP port:

```text
-> {"id": 1, "token": "...", "method": "place_order",
    "params": {"ticker": "INFY", "side": "buy", "quantity": 5, "order_type": "market"}}
<- {"id": 1, "result": {"order_id": "...", "status": "filled", ...}}
<- {"id": 2, "error": {"type": "OrderRejected", "message": "..."}}
```

| Method | Params | Result |
|---|---|---|
| `health` | — | `{backend, orders_enabled, account, version}` |
| `get_account_balance` | — | `{cash, equity, buying_power}` |
| `get_positions` | — | `{ticker: {ticker, quantity, avg_cost, market_value, unrealized_pnl}}` |
| `place_order` | `{ticker, side, quantity, order_type, limit_price?}` | Order |
| `get_order_status` / `cancel_order` | `{order_id}` | Order / null |
| `get_historical_bars` | `{ticker, start, end, timeframe}` | `[Bar]` (kite only) |
| `mark_price` | `{ticker, price}` | null (paper only) |

Error types: `OrderRejected`, `AuthExpired`, `Unauthorized`,
`InvalidRequest`, `NotSupported`, `Broker`. The data shapes match the
Python contracts in `src/trading_agents/contracts/`.

## Development

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test                # unit + integration (mock Kite API, real socket)
cargo build               # then, from the repo root, the cross-language tests:
..\.venv\Scripts\pytest tests\integration\test_execd_roundtrip.py
```
