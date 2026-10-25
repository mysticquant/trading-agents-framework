//! Method dispatch onto the configured broker backend.

use chrono::NaiveDate;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::ServiceError;
use crate::kite::KiteBackend;
use crate::paper::PaperBackend;
use crate::protocol::{OrderRequest, Timeframe, params};

pub enum Backend {
    Paper(PaperBackend),
    // Boxed: the Kite backend (HTTP client, caches) is much larger.
    Kite(Box<KiteBackend>),
}

#[derive(Deserialize)]
struct OrderId {
    order_id: String,
}

#[derive(Deserialize)]
struct BarsParams {
    ticker: String,
    start: NaiveDate,
    end: NaiveDate,
    timeframe: Timeframe,
}

#[derive(Deserialize)]
struct MarkPrice {
    ticker: String,
    price: f64,
}

fn to_value<T: serde::Serialize>(value: T) -> Result<Value, ServiceError> {
    serde_json::to_value(value).map_err(|e| ServiceError::Broker(format!("encoding response: {e}")))
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Paper(_) => "paper",
            Backend::Kite(_) => "kite",
        }
    }

    pub fn orders_enabled(&self) -> bool {
        match self {
            Backend::Paper(_) => true,
            Backend::Kite(kite) => kite.orders_enabled(),
        }
    }

    /// Validate the broker session at startup, so a bad token fails fast.
    pub async fn connect(&self) -> Result<(), ServiceError> {
        match self {
            Backend::Paper(_) => Ok(()),
            Backend::Kite(kite) => kite.connect().await,
        }
    }

    pub async fn handle(&self, method: &str, raw_params: Value) -> Result<Value, ServiceError> {
        match (method, self) {
            ("health", _) => {
                let account = match self {
                    Backend::Kite(kite) => kite.user_id().await,
                    Backend::Paper(_) => None,
                };
                Ok(json!({
                    "backend": self.name(),
                    "orders_enabled": self.orders_enabled(),
                    "account": account,
                    "version": env!("CARGO_PKG_VERSION"),
                }))
            }
            ("get_account_balance", Backend::Paper(paper)) => to_value(paper.balance().await),
            ("get_account_balance", Backend::Kite(kite)) => to_value(kite.balance().await?),
            ("get_positions", Backend::Paper(paper)) => to_value(paper.positions().await),
            ("get_positions", Backend::Kite(kite)) => to_value(kite.positions().await?),
            ("place_order", _) => {
                let request: OrderRequest = params(raw_params)?;
                let result = match self {
                    Backend::Paper(paper) => paper.place_order(&request).await,
                    Backend::Kite(kite) => kite.place_order(&request).await,
                };
                match &result {
                    Ok(order) => tracing::info!(
                        order_id = %order.order_id, ticker = %order.ticker, side = ?order.side,
                        quantity = order.quantity, status = ?order.status, "order placed"
                    ),
                    Err(err) => {
                        tracing::warn!(ticker = %request.ticker, error = %err, "order refused")
                    }
                }
                to_value(result?)
            }
            ("get_order_status", _) => {
                let OrderId { order_id } = params(raw_params)?;
                match self {
                    Backend::Paper(paper) => to_value(paper.order_status(&order_id).await?),
                    Backend::Kite(kite) => to_value(kite.order_status(&order_id).await?),
                }
            }
            ("cancel_order", _) => {
                let OrderId { order_id } = params(raw_params)?;
                match self {
                    Backend::Paper(paper) => paper.cancel_order(&order_id).await?,
                    Backend::Kite(kite) => kite.cancel_order(&order_id).await?,
                }
                Ok(Value::Null)
            }
            ("get_historical_bars", Backend::Kite(kite)) => {
                let p: BarsParams = params(raw_params)?;
                to_value(
                    kite.historical_bars(&p.ticker, p.start, p.end, p.timeframe)
                        .await?,
                )
            }
            ("get_historical_bars", Backend::Paper(_)) => Err(ServiceError::NotSupported(
                "the paper backend has no market data; use a HistoricalDataProvider".into(),
            )),
            ("mark_price", Backend::Paper(paper)) => {
                let p: MarkPrice = params(raw_params)?;
                paper.mark_price(&p.ticker, p.price).await?;
                Ok(Value::Null)
            }
            ("mark_price", Backend::Kite(_)) => Err(ServiceError::NotSupported(
                "mark_price is only for the paper backend; Kite knows its own prices".into(),
            )),
            (other, _) => Err(ServiceError::InvalidRequest(format!(
                "unknown method {other:?}"
            ))),
        }
    }
}
