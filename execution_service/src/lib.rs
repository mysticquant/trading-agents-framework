//! `execd`: a local order-execution service for trading-agents.
//!
//! Exposes the Python `BrokerAdapter` operations (balance, positions,
//! orders, history) over newline-delimited JSON on a loopback TCP port,
//! backed by a simulated ledger (`paper`) or Zerodha Kite (`kite`). Python
//! talks to it through `trading_agents.brokers.execution_service`.

pub mod backend;
pub mod error;
pub mod kite;
pub mod paper;
pub mod protocol;
pub mod server;
