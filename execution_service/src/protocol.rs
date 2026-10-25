//! Wire protocol: one JSON object per line in each direction.
//!
//! Request:  `{"id": 1, "token": "...", "method": "place_order", "params": {...}}`
//! Response: `{"id": 1, "result": ...}` or
//!           `{"id": 1, "error": {"type": "OrderRejected", "message": "..."}}`
//!
//! The data types mirror the Python contracts in
//! `trading_agents/contracts/broker.py` and `contracts/market.py`, field for
//! field, so the Python proxy adapter can validate responses directly into
//! them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ServiceError;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    Pending,
    Submitted,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Timeframe {
    #[serde(rename = "1d")]
    Day,
    #[serde(rename = "1h")]
    Hour,
    #[serde(rename = "5m")]
    FiveMinutes,
}

impl Timeframe {
    pub fn as_str(self) -> &'static str {
        match self {
            Timeframe::Day => "1d",
            Timeframe::Hour => "1h",
            Timeframe::FiveMinutes => "5m",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrderRequest {
    pub ticker: String,
    pub side: Side,
    pub quantity: f64,
    pub order_type: OrderType,
    #[serde(default)]
    pub limit_price: Option<f64>,
}

impl OrderRequest {
    /// Brokers take whole shares; fractional requests are floored and an
    /// order that floors to zero is refused rather than silently changed.
    pub fn whole_shares(&self) -> Result<u64, ServiceError> {
        if !self.quantity.is_finite() || self.quantity < 1.0 {
            return Err(ServiceError::OrderRejected(format!(
                "{} shares of {} rounds down to 0 whole shares",
                self.quantity, self.ticker
            )));
        }
        Ok(self.quantity.floor() as u64)
    }

    pub fn limit_price(&self) -> Result<Option<f64>, ServiceError> {
        match (self.order_type, self.limit_price) {
            (OrderType::Limit, None) => Err(ServiceError::OrderRejected(
                "limit orders need a limit_price".into(),
            )),
            (OrderType::Limit, Some(price)) if !(price.is_finite() && price > 0.0) => Err(
                ServiceError::OrderRejected(format!("invalid limit_price {price}")),
            ),
            (OrderType::Limit, price) => Ok(price),
            (OrderType::Market, _) => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Order {
    pub order_id: String,
    pub ticker: String,
    pub side: Side,
    pub quantity: f64,
    pub order_type: OrderType,
    pub limit_price: Option<f64>,
    pub status: OrderStatus,
    pub submitted_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub fill_price: Option<f64>,
    pub commission: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub ticker: String,
    pub quantity: f64,
    pub avg_cost: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AccountBalance {
    pub cash: f64,
    pub equity: f64,
    pub buying_power: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub ticker: String,
    /// ISO 8601. Daily bars are naive trading-date midnights
    /// ("2024-01-02T00:00:00"); intraday bars carry their UTC offset.
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timeframe: Timeframe,
}

#[derive(Debug, Deserialize)]
pub struct Request {
    pub id: u64,
    #[serde(default)]
    pub token: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ErrorBody {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBody>,
}

impl Response {
    pub fn from_result(id: u64, result: Result<Value, ServiceError>) -> Self {
        match result {
            Ok(value) => Response {
                id,
                result: Some(value),
                error: None,
            },
            Err(err) => Response {
                id,
                result: None,
                error: Some(ErrorBody {
                    kind: err.kind().to_string(),
                    message: err.to_string(),
                }),
            },
        }
    }
}

/// Decode a method's `params` into its typed form.
pub fn params<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, ServiceError> {
    serde_json::from_value(value)
        .map_err(|e| ServiceError::InvalidRequest(format!("bad params: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(quantity: f64, order_type: OrderType, limit_price: Option<f64>) -> OrderRequest {
        OrderRequest {
            ticker: "INFY".into(),
            side: Side::Buy,
            quantity,
            order_type,
            limit_price,
        }
    }

    #[test]
    fn floors_to_whole_shares_and_refuses_zero() {
        assert_eq!(
            order(9.7, OrderType::Market, None).whole_shares().unwrap(),
            9
        );
        assert!(matches!(
            order(0.9, OrderType::Market, None).whole_shares(),
            Err(ServiceError::OrderRejected(_))
        ));
        assert!(
            order(f64::NAN, OrderType::Market, None)
                .whole_shares()
                .is_err()
        );
    }

    #[test]
    fn limit_orders_need_a_positive_price() {
        assert!(order(1.0, OrderType::Limit, None).limit_price().is_err());
        assert!(
            order(1.0, OrderType::Limit, Some(-1.0))
                .limit_price()
                .is_err()
        );
        assert_eq!(
            order(1.0, OrderType::Limit, Some(10.5))
                .limit_price()
                .unwrap(),
            Some(10.5)
        );
        assert_eq!(
            order(1.0, OrderType::Market, Some(10.5))
                .limit_price()
                .unwrap(),
            None
        );
    }

    #[test]
    fn error_responses_carry_a_type_and_message() {
        let response = Response::from_result(7, Err(ServiceError::AuthExpired("expired".into())));
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"id": 7, "error": {"type": "AuthExpired", "message": "expired"}})
        );
    }

    #[test]
    fn statuses_and_timeframes_match_the_python_contracts() {
        assert_eq!(
            serde_json::to_value(OrderStatus::PartiallyFilled).unwrap(),
            "partially_filled"
        );
        assert_eq!(serde_json::to_value(Timeframe::FiveMinutes).unwrap(), "5m");
        assert_eq!(serde_json::to_value(Side::Sell).unwrap(), "sell");
    }
}
