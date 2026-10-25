//! Simulated broker: an in-memory cash/positions ledger with immediate
//! fills at the last marked price plus fixed slippage and commission. Same
//! fill model as the Python `PaperBrokerAdapter`, except quantities are
//! whole shares like the real brokers this service fronts.
//!
//! Prices come from `mark_price` calls (the client marks each ticker
//! before ordering), standing in for the last trade a real broker knows.

use std::collections::{BTreeMap, HashMap};

use chrono::Utc;
use tokio::sync::Mutex;

use crate::error::ServiceError;
use crate::protocol::{
    AccountBalance, Order, OrderRequest, OrderStatus, OrderType, Position, Side,
};

#[derive(Debug, Clone, Copy)]
struct Holding {
    quantity: f64,
    avg_cost: f64,
}

#[derive(Debug)]
struct Ledger {
    cash: f64,
    holdings: BTreeMap<String, Holding>,
    last_price: HashMap<String, f64>,
    orders: HashMap<String, Order>,
    next_order_id: u64,
}

pub struct PaperBackend {
    commission_bps: f64,
    slippage_bps: f64,
    ledger: Mutex<Ledger>,
}

impl PaperBackend {
    pub fn new(initial_cash: f64, commission_bps: f64, slippage_bps: f64) -> Self {
        PaperBackend {
            commission_bps,
            slippage_bps,
            ledger: Mutex::new(Ledger {
                cash: initial_cash,
                holdings: BTreeMap::new(),
                last_price: HashMap::new(),
                orders: HashMap::new(),
                next_order_id: 1,
            }),
        }
    }

    pub async fn mark_price(&self, ticker: &str, price: f64) -> Result<(), ServiceError> {
        if !(price.is_finite() && price > 0.0) {
            return Err(ServiceError::InvalidRequest(format!(
                "invalid price {price}"
            )));
        }
        self.ledger
            .lock()
            .await
            .last_price
            .insert(ticker.to_string(), price);
        Ok(())
    }

    pub async fn positions(&self) -> BTreeMap<String, Position> {
        let ledger = self.ledger.lock().await;
        ledger
            .holdings
            .iter()
            .map(|(ticker, h)| {
                let mark = ledger.last_price.get(ticker).copied().unwrap_or(h.avg_cost);
                let market_value = h.quantity * mark;
                (
                    ticker.clone(),
                    Position {
                        ticker: ticker.clone(),
                        quantity: h.quantity,
                        avg_cost: h.avg_cost,
                        market_value,
                        unrealized_pnl: market_value - h.quantity * h.avg_cost,
                    },
                )
            })
            .collect()
    }

    pub async fn balance(&self) -> AccountBalance {
        let positions_value: f64 = self
            .positions()
            .await
            .values()
            .map(|p| p.market_value)
            .sum();
        let cash = self.ledger.lock().await.cash;
        AccountBalance {
            cash,
            equity: cash + positions_value,
            buying_power: cash,
        }
    }

    pub async fn place_order(&self, request: &OrderRequest) -> Result<Order, ServiceError> {
        let quantity = request.whole_shares()? as f64;
        let limit_price = request.limit_price()?;
        let mut ledger = self.ledger.lock().await;
        let mark = *ledger.last_price.get(&request.ticker).ok_or_else(|| {
            ServiceError::OrderRejected(format!(
                "no current price for {}; call mark_price first",
                request.ticker
            ))
        })?;

        let reference = limit_price.unwrap_or(mark);
        let slippage = self.slippage_bps / 10_000.0;
        let fill_price = match request.side {
            Side::Buy => reference * (1.0 + slippage),
            Side::Sell => reference * (1.0 - slippage),
        };
        let commission = fill_price * quantity * self.commission_bps / 10_000.0;

        match request.side {
            Side::Buy => {
                let cost = fill_price * quantity + commission;
                if cost > ledger.cash {
                    return Err(ServiceError::OrderRejected(format!(
                        "insufficient cash for {}: need {cost:.2}, have {:.2}",
                        request.ticker, ledger.cash
                    )));
                }
                ledger.cash -= cost;
                let holding = ledger
                    .holdings
                    .entry(request.ticker.clone())
                    .or_insert(Holding {
                        quantity: 0.0,
                        avg_cost: 0.0,
                    });
                let new_quantity = holding.quantity + quantity;
                holding.avg_cost =
                    (holding.quantity * holding.avg_cost + quantity * fill_price) / new_quantity;
                holding.quantity = new_quantity;
            }
            Side::Sell => {
                let held = ledger
                    .holdings
                    .get(&request.ticker)
                    .map_or(0.0, |h| h.quantity);
                if quantity > held {
                    return Err(ServiceError::OrderRejected(format!(
                        "cannot sell {quantity} of {}; only hold {held}",
                        request.ticker
                    )));
                }
                ledger.cash += fill_price * quantity - commission;
                if quantity == held {
                    ledger.holdings.remove(&request.ticker);
                } else if let Some(holding) = ledger.holdings.get_mut(&request.ticker) {
                    holding.quantity -= quantity;
                }
            }
        }

        let now = Utc::now();
        let order = Order {
            order_id: ledger.next_order_id.to_string(),
            ticker: request.ticker.clone(),
            side: request.side,
            quantity,
            order_type: request.order_type,
            limit_price: if request.order_type == OrderType::Limit {
                limit_price
            } else {
                None
            },
            status: OrderStatus::Filled,
            submitted_at: now,
            filled_at: Some(now),
            fill_price: Some(fill_price),
            commission,
        };
        ledger.next_order_id += 1;
        ledger.orders.insert(order.order_id.clone(), order.clone());
        Ok(order)
    }

    pub async fn order_status(&self, order_id: &str) -> Result<Order, ServiceError> {
        self.ledger
            .lock()
            .await
            .orders
            .get(order_id)
            .cloned()
            .ok_or_else(|| ServiceError::Broker(format!("unknown order_id: {order_id}")))
    }

    /// Orders fill immediately, so there is never anything left to cancel.
    pub async fn cancel_order(&self, order_id: &str) -> Result<(), ServiceError> {
        self.order_status(order_id).await.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(side: Side, quantity: f64) -> OrderRequest {
        OrderRequest {
            ticker: "AAPL".into(),
            side,
            quantity,
            order_type: OrderType::Market,
            limit_price: None,
        }
    }

    #[tokio::test]
    async fn buy_then_sell_round_trip_with_costs() {
        let broker = PaperBackend::new(10_000.0, 10.0, 10.0);
        broker.mark_price("AAPL", 100.0).await.unwrap();

        let buy = broker.place_order(&request(Side::Buy, 10.6)).await.unwrap();
        assert_eq!(buy.quantity, 10.0); // floored
        assert!((buy.fill_price.unwrap() - 100.1).abs() < 1e-9);
        assert!((buy.commission - 1.001).abs() < 1e-9);
        let balance = broker.balance().await;
        assert!((balance.cash - (10_000.0 - 1001.0 - 1.001)).abs() < 1e-6);

        broker
            .place_order(&request(Side::Sell, 10.0))
            .await
            .unwrap();
        assert!(broker.positions().await.is_empty());
    }

    #[tokio::test]
    async fn refuses_unaffordable_buys_and_oversized_sells() {
        let broker = PaperBackend::new(500.0, 0.0, 0.0);
        broker.mark_price("AAPL", 100.0).await.unwrap();

        assert!(matches!(
            broker.place_order(&request(Side::Buy, 6.0)).await,
            Err(ServiceError::OrderRejected(m)) if m.contains("insufficient cash")
        ));
        assert!(matches!(
            broker.place_order(&request(Side::Sell, 1.0)).await,
            Err(ServiceError::OrderRejected(m)) if m.contains("only hold 0")
        ));
    }

    #[tokio::test]
    async fn needs_a_marked_price() {
        let broker = PaperBackend::new(500.0, 0.0, 0.0);
        let err = broker
            .place_order(&request(Side::Buy, 1.0))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("mark_price"));
    }

    #[tokio::test]
    async fn averages_cost_across_buys_and_marks_to_market() {
        let broker = PaperBackend::new(100_000.0, 0.0, 0.0);
        broker.mark_price("AAPL", 100.0).await.unwrap();
        broker.place_order(&request(Side::Buy, 10.0)).await.unwrap();
        broker.mark_price("AAPL", 200.0).await.unwrap();
        broker.place_order(&request(Side::Buy, 10.0)).await.unwrap();

        let position = broker.positions().await["AAPL"].clone();
        assert_eq!(position.avg_cost, 150.0);
        assert_eq!(position.market_value, 4000.0);
        assert_eq!(position.unrealized_pnl, 1000.0);
    }
}
