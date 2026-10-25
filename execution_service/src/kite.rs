//! Zerodha Kite Connect v3 backend over its REST API — a port of the Python
//! `trading_agents/brokers/kite.py` adapter with the same rules:
//!
//! - Kite has no paper trading, so orders are refused unless the service
//!   was started with `--allow-live`. Reads always work.
//! - Market orders carry `market_protection=-1` (required for API market
//!   orders); quantities are whole shares; the product defaults to CNC.
//! - Orders are only accepted from the static IP registered with Zerodha
//!   (SEBI retail algo rules); a refusal mentioning the IP says so.
//! - Expired or invalid sessions (6 AM IST daily expiry) map to AuthExpired.
//! - Requests are throttled to Kite's published limits (historical 3/s,
//!   everything else 10/s), over one persistent HTTPS connection pool.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::time::Duration;

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Asia::Kolkata;
use reqwest::Method;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::error::ServiceError;
use crate::protocol::{
    AccountBalance, Bar, Order, OrderRequest, OrderStatus, OrderType, Position, Side, Timeframe,
};

pub const API_BASE: &str = "https://api.kite.trade";

const INTERIM_STATUSES: &[&str] = &[
    "PUT ORDER REQ RECEIVED",
    "VALIDATION PENDING",
    "OPEN PENDING",
    "MODIFY VALIDATION PENDING",
    "MODIFY PENDING",
    "CANCEL PENDING",
];
/// Kite error types meaning the order itself was refused.
const ORDER_REFUSALS: &[&str] = &[
    "InputException",
    "OrderException",
    "MarginException",
    "HoldingException",
];

#[derive(Debug, Clone)]
pub struct KiteConfig {
    pub api_key: String,
    pub access_token: String,
    pub base_url: String,
    pub exchange: String,
    pub product: String,
    pub allow_live: bool,
    pub ack_timeout: Duration,
    pub poll_interval: Duration,
    /// Minimum spacing between historical / other requests.
    pub min_interval_historical: Duration,
    pub min_interval_default: Duration,
}

impl KiteConfig {
    pub fn new(api_key: impl Into<String>, access_token: impl Into<String>) -> Self {
        KiteConfig {
            api_key: api_key.into(),
            access_token: access_token.into(),
            base_url: API_BASE.into(),
            exchange: "NSE".into(),
            product: "CNC".into(),
            allow_live: false,
            ack_timeout: Duration::from_secs(10),
            poll_interval: Duration::from_millis(500),
            min_interval_historical: Duration::from_millis(334),
            min_interval_default: Duration::from_millis(100),
        }
    }
}

/// A Kite API error before it is classified for the caller.
#[derive(Debug)]
struct KiteFailure {
    error_type: String,
    status: u16,
    message: String,
}

enum RequestError {
    Api(KiteFailure),
    Service(ServiceError),
}

impl From<RequestError> for ServiceError {
    fn from(err: RequestError) -> Self {
        match err {
            RequestError::Service(e) => e,
            RequestError::Api(f) => {
                ServiceError::Broker(format!("Kite {}: {}", f.error_type, f.message))
            }
        }
    }
}

pub struct KiteBackend {
    config: KiteConfig,
    http: reqwest::Client,
    last_request: Mutex<HashMap<&'static str, Instant>>,
    instruments: Mutex<Option<(NaiveDate, HashMap<String, u64>)>>,
    user_id: Mutex<Option<String>>,
}

impl KiteBackend {
    pub fn new(config: KiteConfig) -> Result<Self, ServiceError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| ServiceError::Broker(format!("could not build HTTP client: {e}")))?;
        Ok(KiteBackend {
            config,
            http,
            last_request: Mutex::new(HashMap::new()),
            instruments: Mutex::new(None),
            user_id: Mutex::new(None),
        })
    }

    pub fn orders_enabled(&self) -> bool {
        self.config.allow_live
    }

    pub async fn user_id(&self) -> Option<String> {
        self.user_id.lock().await.clone()
    }

    async fn throttle(&self, category: &'static str) {
        let min_interval = if category == "historical" {
            self.config.min_interval_historical
        } else {
            self.config.min_interval_default
        };
        let mut last = self.last_request.lock().await;
        if let Some(previous) = last.get(category) {
            let ready_at = *previous + min_interval;
            if ready_at > Instant::now() {
                tokio::time::sleep_until(ready_at).await;
            }
        }
        last.insert(category, Instant::now());
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        query: &[(&str, String)],
        form: Option<&[(&str, String)]>,
    ) -> Result<(u16, Vec<u8>), RequestError> {
        let category = if path.starts_with("/instruments/historical") {
            "historical"
        } else {
            "default"
        };
        self.throttle(category).await;

        let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), path);
        let mut builder = self
            .http
            .request(method, url)
            .header("X-Kite-Version", "3")
            .header(
                "Authorization",
                format!("token {}:{}", self.config.api_key, self.config.access_token),
            );
        if !query.is_empty() {
            builder = builder.query(query);
        }
        if let Some(form) = form {
            builder = builder.form(form);
        }
        let response = builder.send().await.map_err(|e| {
            RequestError::Service(ServiceError::Broker(format!("could not reach Kite: {e}")))
        })?;
        let status = response.status().as_u16();
        let body = response.bytes().await.map_err(|e| {
            RequestError::Service(ServiceError::Broker(format!("reading Kite response: {e}")))
        })?;
        Ok((status, body.to_vec()))
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        query: &[(&str, String)],
        form: Option<&[(&str, String)]>,
    ) -> Result<Value, RequestError> {
        let (status, body) = self.send(method, path, query, form).await?;
        let decoded: Value = serde_json::from_slice(if body.is_empty() { b"{}" } else { &body })
            .map_err(|_| {
                RequestError::Service(ServiceError::Broker(format!(
                    "Kite returned HTTP {status} with a non-JSON body"
                )))
            })?;
        if status == 200 && decoded["status"] == "success" {
            return Ok(decoded["data"].clone());
        }

        let error_type = decoded["error_type"]
            .as_str()
            .unwrap_or("GeneralException")
            .to_string();
        let mut message = decoded["message"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("HTTP {status}"));
        if error_type == "TokenException"
            || (status == 403 && message.to_lowercase().contains("token"))
        {
            return Err(RequestError::Service(ServiceError::AuthExpired(format!(
                "Kite session expired or invalid ({message}). Run `trading-agents kite-login` \
                 and restart the service with the new KITE_ACCESS_TOKEN (tokens expire at 6 AM IST)."
            ))));
        }
        if status == 429 {
            message.push_str(" (rate limited)");
        }
        Err(RequestError::Api(KiteFailure {
            error_type,
            status,
            message,
        }))
    }

    async fn get(&self, path: &str) -> Result<Value, ServiceError> {
        Ok(self.request(Method::GET, path, &[], None).await?)
    }

    pub async fn connect(&self) -> Result<(), ServiceError> {
        if self.config.access_token.is_empty() {
            return Err(ServiceError::AuthExpired(
                "no Kite access token; run `trading-agents kite-login` first".into(),
            ));
        }
        let profile = self.get("/user/profile").await?;
        *self.user_id.lock().await = profile["user_id"].as_str().map(str::to_string);
        Ok(())
    }

    pub async fn positions(&self) -> Result<BTreeMap<String, Position>, ServiceError> {
        let holdings = self.get("/portfolio/holdings").await?;
        let positions = self.get("/portfolio/positions").await?;
        Ok(merge_positions(
            &holdings,
            &positions,
            &self.config.exchange,
        ))
    }

    pub async fn balance(&self) -> Result<AccountBalance, ServiceError> {
        let margins = self.get("/user/margins/equity").await?;
        let cash = number(&margins["available"]["live_balance"]);
        let positions_value: f64 = self
            .positions()
            .await?
            .values()
            .map(|p| p.market_value)
            .sum();
        Ok(AccountBalance {
            cash,
            equity: cash + positions_value,
            buying_power: number(&margins["net"]),
        })
    }

    pub async fn place_order(&self, request: &OrderRequest) -> Result<Order, ServiceError> {
        if !self.config.allow_live {
            return Err(ServiceError::OrderRejected(
                "Kite has no paper trading, so every order is real money; the service was \
                 started without --allow-live, so orders are disabled"
                    .into(),
            ));
        }
        let quantity = request.whole_shares()?;
        let limit_price = request.limit_price()?;
        let mut form: Vec<(&str, String)> = vec![
            ("tradingsymbol", request.ticker.to_uppercase()),
            ("exchange", self.config.exchange.clone()),
            (
                "transaction_type",
                match request.side {
                    Side::Buy => "BUY".into(),
                    Side::Sell => "SELL".into(),
                },
            ),
            ("quantity", quantity.to_string()),
            ("product", self.config.product.clone()),
            ("validity", "DAY".into()),
            ("tag", "tradingagents".into()),
        ];
        match limit_price {
            Some(price) => {
                form.push(("order_type", "LIMIT".into()));
                form.push(("price", price.to_string()));
            }
            None => {
                form.push(("order_type", "MARKET".into()));
                // Required for API market orders; -1 = Kite's automatic protection.
                form.push(("market_protection", "-1".into()));
            }
        }

        let placed = match self
            .request(Method::POST, "/orders/regular", &[], Some(&form))
            .await
        {
            Ok(data) => data,
            Err(RequestError::Api(failure))
                if ORDER_REFUSALS.contains(&failure.error_type.as_str())
                    || failure.status == 400
                    || failure.status == 403 =>
            {
                let hint = if mentions_ip(&failure.message) {
                    " Kite only accepts API orders from your registered static IP."
                } else {
                    ""
                };
                return Err(ServiceError::OrderRejected(format!(
                    "Kite refused the order: {}.{hint}",
                    failure.message
                )));
            }
            Err(other) => return Err(other.into()),
        };
        let order_id = placed["order_id"]
            .as_str()
            .map(str::to_string)
            .or_else(|| placed["order_id"].as_u64().map(|id| id.to_string()))
            .ok_or_else(|| ServiceError::Broker("Kite returned no order_id".into()))?;

        let record = self.wait_for_ack(&order_id).await?;
        let order = record_to_order(&record)?;
        if matches!(order.status, OrderStatus::Rejected | OrderStatus::Cancelled) {
            let reason = record["status_message"]
                .as_str()
                .unwrap_or("no reason given");
            let status = if order.status == OrderStatus::Rejected {
                "rejected"
            } else {
                "cancelled"
            };
            return Err(ServiceError::OrderRejected(format!(
                "Kite {status} the order for {quantity} {}: {reason}",
                request.ticker
            )));
        }
        Ok(order)
    }

    /// Poll the order until it leaves Kite's interim states or the ack
    /// timeout passes; returns the latest record either way.
    async fn wait_for_ack(&self, order_id: &str) -> Result<Value, ServiceError> {
        let deadline = Instant::now() + self.config.ack_timeout;
        loop {
            let record = self.latest_record(order_id).await?;
            if map_status(&record) != OrderStatus::Pending || Instant::now() >= deadline {
                return Ok(record);
            }
            tokio::time::sleep(self.config.poll_interval).await;
        }
    }

    async fn latest_record(&self, order_id: &str) -> Result<Value, ServiceError> {
        let history = self.get(&format!("/orders/{order_id}")).await?;
        history
            .as_array()
            .and_then(|records| records.last())
            .cloned()
            .ok_or_else(|| ServiceError::Broker(format!("unknown order_id: {order_id}")))
    }

    pub async fn order_status(&self, order_id: &str) -> Result<Order, ServiceError> {
        record_to_order(&self.latest_record(order_id).await?)
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<(), ServiceError> {
        let record = self.latest_record(order_id).await?;
        if matches!(
            map_status(&record),
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected
        ) {
            return Ok(());
        }
        self.request(
            Method::DELETE,
            &format!("/orders/regular/{order_id}"),
            &[],
            None,
        )
        .await?;
        Ok(())
    }

    async fn instrument_token(&self, ticker: &str) -> Result<u64, ServiceError> {
        let today = Utc::now().with_timezone(&Kolkata).date_naive();
        let mut cache = self.instruments.lock().await;
        // The instrument dump is regenerated daily; reload once a day.
        if !matches!(&*cache, Some((loaded_on, _)) if *loaded_on == today) {
            let path = format!("/instruments/{}", self.config.exchange);
            let (status, body) = self
                .send(Method::GET, &path, &[], None)
                .await
                .map_err(ServiceError::from)?;
            if status != 200 {
                return Err(ServiceError::Broker(format!(
                    "instrument dump returned HTTP {status}"
                )));
            }
            *cache = Some((
                today,
                parse_instruments(&gunzip_if_needed(body)?, &self.config.exchange)?,
            ));
        }
        let (_, tokens) = cache.as_ref().expect("instrument cache was just filled");
        tokens.get(&ticker.to_uppercase()).copied().ok_or_else(|| {
            ServiceError::Broker(format!(
                "{ticker} is not a {} equity instrument on Kite",
                self.config.exchange
            ))
        })
    }

    pub async fn historical_bars(
        &self,
        ticker: &str,
        start: NaiveDate,
        end: NaiveDate,
        timeframe: Timeframe,
    ) -> Result<Vec<Bar>, ServiceError> {
        let token = self.instrument_token(ticker).await?;
        let (interval, max_days) = match timeframe {
            Timeframe::Day => ("day", 2000),
            Timeframe::Hour => ("60minute", 400),
            Timeframe::FiveMinutes => ("5minute", 100),
        };
        let mut bars = Vec::new();
        let mut chunk_start = start;
        while chunk_start <= end {
            let chunk_end = end.min(chunk_start + chrono::Days::new(max_days - 1));
            let data = self
                .request(
                    Method::GET,
                    &format!("/instruments/historical/{token}/{interval}"),
                    &[
                        ("from", format!("{chunk_start} 00:00:00")),
                        ("to", format!("{chunk_end} 23:59:59")),
                    ],
                    None,
                )
                .await?;
            for candle in data["candles"].as_array().into_iter().flatten() {
                bars.push(candle_to_bar(ticker, timeframe, candle)?);
            }
            chunk_start = chunk_end + chrono::Days::new(1);
        }
        Ok(bars)
    }
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn mentions_ip(message: &str) -> bool {
    message
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|word| word.eq_ignore_ascii_case("ip"))
}

/// Delivery holdings (settled + T1) plus today's net equity positions on
/// `exchange`; same-day buys appear only in positions and same-day sells as
/// negative position quantities.
pub fn merge_positions(
    holdings: &Value,
    positions: &Value,
    exchange: &str,
) -> BTreeMap<String, Position> {
    let mut quantity: BTreeMap<String, f64> = BTreeMap::new();
    let mut cost: HashMap<String, f64> = HashMap::new();
    let mut last_price: HashMap<String, f64> = HashMap::new();

    for row in holdings.as_array().into_iter().flatten() {
        if row["exchange"] != exchange {
            continue;
        }
        let Some(symbol) = row["tradingsymbol"].as_str() else {
            continue;
        };
        let qty = number(&row["quantity"]) + number(&row["t1_quantity"]);
        *quantity.entry(symbol.into()).or_default() += qty;
        *cost.entry(symbol.into()).or_default() += qty * number(&row["average_price"]);
        last_price.insert(symbol.into(), number(&row["last_price"]));
    }
    for row in positions["net"].as_array().into_iter().flatten() {
        let product = row["product"].as_str().unwrap_or_default();
        if row["exchange"] != exchange || !matches!(product, "CNC" | "MIS") {
            continue;
        }
        let Some(symbol) = row["tradingsymbol"].as_str() else {
            continue;
        };
        let qty = number(&row["quantity"]);
        *quantity.entry(symbol.into()).or_default() += qty;
        if qty > 0.0 {
            *cost.entry(symbol.into()).or_default() += qty * number(&row["average_price"]);
        }
        let price = number(&row["last_price"]);
        if price > 0.0 || !last_price.contains_key(symbol) {
            last_price.insert(symbol.into(), price);
        }
    }

    quantity
        .into_iter()
        .filter(|(_, qty)| *qty > 0.0)
        .map(|(symbol, qty)| {
            let avg_cost = cost.get(&symbol).copied().unwrap_or(0.0) / qty;
            let market_value = qty * last_price.get(&symbol).copied().unwrap_or(0.0);
            let position = Position {
                ticker: symbol.clone(),
                quantity: qty,
                avg_cost,
                market_value,
                unrealized_pnl: market_value - qty * avg_cost,
            };
            (symbol, position)
        })
        .collect()
}

pub fn map_status(record: &Value) -> OrderStatus {
    let status = record["status"].as_str().unwrap_or_default().to_uppercase();
    let filled = number(&record["filled_quantity"]);
    match status.as_str() {
        "COMPLETE" => OrderStatus::Filled,
        "REJECTED" => OrderStatus::Rejected,
        "CANCELLED" if filled > 0.0 => OrderStatus::PartiallyFilled,
        "CANCELLED" => OrderStatus::Cancelled,
        "" => OrderStatus::Pending,
        s if INTERIM_STATUSES.contains(&s) => OrderStatus::Pending,
        _ if filled > 0.0 => OrderStatus::PartiallyFilled,
        _ => OrderStatus::Submitted,
    }
}

/// Order timestamps are "YYYY-MM-DD HH:MM:SS" in IST.
fn parse_kite_time(value: &Value) -> Option<DateTime<Utc>> {
    let naive = NaiveDateTime::parse_from_str(value.as_str()?, "%Y-%m-%d %H:%M:%S").ok()?;
    Kolkata
        .from_local_datetime(&naive)
        .single()
        .map(|ist| ist.with_timezone(&Utc))
}

pub fn record_to_order(record: &Value) -> Result<Order, ServiceError> {
    let status = map_status(record);
    let order_id = record["order_id"]
        .as_str()
        .map(str::to_string)
        .or_else(|| record["order_id"].as_u64().map(|id| id.to_string()))
        .ok_or_else(|| ServiceError::Broker("order record without order_id".into()))?;
    let is_limit = record["order_type"] == "LIMIT";
    let filled = number(&record["filled_quantity"]);
    let average_price = number(&record["average_price"]);
    let submitted_at = parse_kite_time(&record["order_timestamp"]).unwrap_or_else(Utc::now);
    Ok(Order {
        order_id,
        ticker: record["tradingsymbol"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        side: if record["transaction_type"] == "BUY" {
            Side::Buy
        } else {
            Side::Sell
        },
        quantity: number(&record["quantity"]),
        order_type: if is_limit {
            OrderType::Limit
        } else {
            OrderType::Market
        },
        limit_price: is_limit.then(|| number(&record["price"])),
        status,
        submitted_at,
        filled_at: (status == OrderStatus::Filled)
            .then(|| parse_kite_time(&record["exchange_update_timestamp"]).unwrap_or(submitted_at)),
        fill_price: (filled > 0.0 && average_price > 0.0).then_some(average_price),
        // Kite's order APIs don't report charges (brokerage/STT/fees).
        commission: 0.0,
    })
}

fn gunzip_if_needed(body: Vec<u8>) -> Result<Vec<u8>, ServiceError> {
    if body.starts_with(&[0x1f, 0x8b]) {
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(body.as_slice())
            .read_to_end(&mut decoded)
            .map_err(|e| ServiceError::Broker(format!("bad instrument dump: {e}")))?;
        Ok(decoded)
    } else {
        Ok(body)
    }
}

/// Trading symbol -> instrument token for plain equities ("EQ").
pub fn parse_instruments(
    dump: &[u8],
    exchange: &str,
) -> Result<HashMap<String, u64>, ServiceError> {
    let mut reader = csv::Reader::from_reader(dump);
    let headers = reader
        .headers()
        .map_err(|e| ServiceError::Broker(format!("bad instrument dump: {e}")))?
        .clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| ServiceError::Broker(format!("instrument dump has no {name} column")))
    };
    let (token_col, symbol_col, type_col, exchange_col) = (
        column("instrument_token")?,
        column("tradingsymbol")?,
        column("instrument_type")?,
        column("exchange")?,
    );
    let mut tokens = HashMap::new();
    for row in reader.records() {
        let row = row.map_err(|e| ServiceError::Broker(format!("bad instrument dump: {e}")))?;
        if row.get(exchange_col) == Some(exchange)
            && row.get(type_col) == Some("EQ")
            && let (Some(symbol), Some(Ok(token))) = (
                row.get(symbol_col),
                row.get(token_col).map(str::parse::<u64>),
            )
        {
            tokens.insert(symbol.to_uppercase(), token);
        }
    }
    Ok(tokens)
}

fn candle_to_bar(ticker: &str, timeframe: Timeframe, candle: &Value) -> Result<Bar, ServiceError> {
    let fields = candle
        .as_array()
        .filter(|f| f.len() >= 6)
        .ok_or_else(|| ServiceError::Broker(format!("malformed candle: {candle}")))?;
    let stamp = fields[0].as_str().unwrap_or_default();
    let moment = DateTime::parse_from_str(stamp, "%Y-%m-%dT%H:%M:%S%z")
        .map_err(|e| ServiceError::Broker(format!("bad candle timestamp {stamp:?}: {e}")))?;
    // Daily bars are keyed by trading date (naive midnight), matching the
    // Python side; intraday bars keep their exact time.
    let timestamp = match timeframe {
        Timeframe::Day => format!("{}T00:00:00", moment.date_naive()),
        _ => moment.with_timezone(&Utc).to_rfc3339(),
    };
    Ok(Bar {
        ticker: ticker.to_string(),
        timestamp,
        open: number(&fields[1]),
        high: number(&fields[2]),
        low: number(&fields[3]),
        close: number(&fields[4]),
        volume: number(&fields[5]),
        timeframe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn statuses_map_like_the_python_adapter() {
        assert_eq!(
            map_status(&json!({"status": "COMPLETE"})),
            OrderStatus::Filled
        );
        assert_eq!(
            map_status(&json!({"status": "VALIDATION PENDING"})),
            OrderStatus::Pending
        );
        assert_eq!(
            map_status(&json!({"status": "OPEN"})),
            OrderStatus::Submitted
        );
        assert_eq!(
            map_status(&json!({"status": "OPEN", "filled_quantity": 3})),
            OrderStatus::PartiallyFilled
        );
        assert_eq!(
            map_status(&json!({"status": "CANCELLED", "filled_quantity": 3})),
            OrderStatus::PartiallyFilled
        );
        assert_eq!(
            map_status(&json!({"status": "REJECTED"})),
            OrderStatus::Rejected
        );
    }

    #[test]
    fn order_times_are_read_as_ist() {
        let order = record_to_order(&json!({
            "order_id": "1", "status": "COMPLETE", "tradingsymbol": "INFY",
            "transaction_type": "BUY", "order_type": "MARKET", "quantity": 9,
            "filled_quantity": 9, "average_price": 1601.5,
            "order_timestamp": "2026-10-26 10:15:00",
            "exchange_update_timestamp": "2026-10-26 10:15:02"
        }))
        .unwrap();
        assert_eq!(
            order.filled_at.unwrap().to_rfc3339(),
            "2026-10-26T04:45:02+00:00"
        );
        assert_eq!(order.fill_price, Some(1601.5));
    }

    #[test]
    fn positions_combine_holdings_and_todays_positions() {
        let holdings = json!([
            {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 10, "t1_quantity": 2,
             "average_price": 1500.0, "last_price": 1600.0},
            {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 5, "t1_quantity": 0,
             "average_price": 3000.0, "last_price": 3100.0},
            {"tradingsymbol": "SBIN", "exchange": "BSE", "quantity": 7, "t1_quantity": 0,
             "average_price": 500.0, "last_price": 550.0}
        ]);
        let positions = json!({"net": [
            {"tradingsymbol": "INFY", "exchange": "NSE", "product": "CNC", "quantity": 3,
             "average_price": 1580.0, "last_price": 1600.0},
            {"tradingsymbol": "TCS", "exchange": "NSE", "product": "CNC", "quantity": -5,
             "average_price": 0, "last_price": 3100.0}
        ]});

        let merged = merge_positions(&holdings, &positions, "NSE");

        assert_eq!(merged.keys().collect::<Vec<_>>(), vec!["INFY"]);
        let infy = &merged["INFY"];
        assert_eq!(infy.quantity, 15.0);
        assert!((infy.avg_cost - (12.0 * 1500.0 + 3.0 * 1580.0) / 15.0).abs() < 1e-9);
        assert_eq!(infy.market_value, 15.0 * 1600.0);
    }

    #[test]
    fn instruments_keep_exchange_equities_only() {
        let dump = b"instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange\n\
408065,1594,INFY,INFOSYS,0,,0,0.05,1,EQ,NSE,NSE\n\
12345,48,INFY25NOVFUT,INFOSYS,0,2025-11-27,0,0.05,400,FUT,NFO-FUT,NFO\n";
        let tokens = parse_instruments(dump, "NSE").unwrap();
        assert_eq!(tokens, HashMap::from([("INFY".to_string(), 408065)]));
    }

    #[test]
    fn candles_become_bars() {
        let daily = candle_to_bar(
            "INFY",
            Timeframe::Day,
            &json!(["2020-01-02T00:00:00+0530", 1, 2, 0.5, 1.5, 100]),
        )
        .unwrap();
        assert_eq!(daily.timestamp, "2020-01-02T00:00:00");
        let intraday = candle_to_bar(
            "INFY",
            Timeframe::FiveMinutes,
            &json!(["2026-10-26T09:15:00+0530", 1, 2, 0.5, 1.5, 100]),
        )
        .unwrap();
        assert_eq!(intraday.timestamp, "2026-10-26T03:45:00+00:00");
    }

    #[test]
    fn ip_mentions_are_whole_words() {
        assert!(mentions_ip("IP not allowed"));
        assert!(mentions_ip("Order from unregistered ip."));
        assert!(!mentions_ip("Shipping delayed"));
    }
}
