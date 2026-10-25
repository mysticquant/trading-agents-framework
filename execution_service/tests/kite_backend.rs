//! The Kite backend against a mock Kite Connect API (wiremock), checking
//! the exact requests sent and how responses and errors are mapped.

use std::time::Duration;

use chrono::NaiveDate;
use execd::error::ServiceError;
use execd::kite::{KiteBackend, KiteConfig};
use execd::protocol::{OrderRequest, OrderStatus, OrderType, Side, Timeframe};
use serde_json::{Value, json};
use wiremock::matchers::{body_string_contains, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ok(data: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({"status": "success", "data": data}))
}

fn kite_error(status: u16, error_type: &str, message: &str) -> ResponseTemplate {
    ResponseTemplate::new(status)
        .set_body_json(json!({"status": "error", "error_type": error_type, "message": message}))
}

fn order_record(status: &str, extra: Value) -> Value {
    let mut record = json!({
        "order_id": "250101000000001", "status": status, "tradingsymbol": "INFY",
        "transaction_type": "BUY", "order_type": "MARKET", "quantity": 9,
        "filled_quantity": 0, "average_price": 0, "price": 0,
        "order_timestamp": "2026-10-26 10:15:00", "status_message": null
    });
    record
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    record
}

async fn backend(server: &MockServer, allow_live: bool) -> KiteBackend {
    let mut config = KiteConfig::new("key", "tok");
    config.base_url = server.uri();
    config.allow_live = allow_live;
    config.ack_timeout = Duration::from_millis(300);
    config.poll_interval = Duration::from_millis(10);
    config.min_interval_default = Duration::ZERO;
    config.min_interval_historical = Duration::ZERO;
    KiteBackend::new(config).unwrap()
}

fn buy(quantity: f64) -> OrderRequest {
    OrderRequest {
        ticker: "infy".into(),
        side: Side::Buy,
        quantity,
        order_type: OrderType::Market,
        limit_price: None,
    }
}

#[tokio::test]
async fn connect_sends_kite_auth_headers_and_reads_the_user() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/user/profile"))
        .and(header("Authorization", "token key:tok"))
        .and(header("X-Kite-Version", "3"))
        .respond_with(ok(json!({"user_id": "AB1234"})))
        .expect(1)
        .mount(&server)
        .await;
    let kite = backend(&server, false).await;

    kite.connect().await.unwrap();

    assert_eq!(kite.user_id().await.as_deref(), Some("AB1234"));
}

#[tokio::test]
async fn expired_token_maps_to_auth_expired() {
    let server = MockServer::start().await;
    Mock::given(path("/user/profile"))
        .respond_with(kite_error(
            403,
            "TokenException",
            "Token is invalid or has expired",
        ))
        .mount(&server)
        .await;

    let err = backend(&server, false).await.connect().await.unwrap_err();

    assert!(matches!(err, ServiceError::AuthExpired(m) if m.contains("kite-login")));
}

#[tokio::test]
async fn orders_are_refused_without_allow_live_and_never_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let err = backend(&server, false)
        .await
        .place_order(&buy(5.0))
        .await
        .unwrap_err();

    assert!(matches!(err, ServiceError::OrderRejected(m) if m.contains("--allow-live")));
}

#[tokio::test]
async fn market_order_sends_market_protection_and_polls_to_the_fill() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/orders/regular"))
        .and(body_string_contains("tradingsymbol=INFY"))
        .and(body_string_contains("quantity=9"))
        .and(body_string_contains("order_type=MARKET"))
        .and(body_string_contains("market_protection=-1"))
        .and(body_string_contains("product=CNC"))
        .and(body_string_contains("validity=DAY"))
        .respond_with(ok(json!({"order_id": "250101000000001"})))
        .expect(1)
        .mount(&server)
        .await;
    // First poll: still interim; afterwards: complete.
    Mock::given(path("/orders/250101000000001"))
        .respond_with(ok(json!([order_record("VALIDATION PENDING", json!({}))])))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(path("/orders/250101000000001"))
        .respond_with(ok(json!([order_record(
            "COMPLETE",
            json!({"filled_quantity": 9, "average_price": 1601.5,
                   "exchange_update_timestamp": "2026-10-26 10:15:02"})
        )])))
        .mount(&server)
        .await;

    let order = backend(&server, true)
        .await
        .place_order(&buy(9.7))
        .await
        .unwrap();

    assert_eq!(order.status, OrderStatus::Filled);
    assert_eq!(order.fill_price, Some(1601.5));
    assert_eq!(
        order.filled_at.unwrap().to_rfc3339(),
        "2026-10-26T04:45:02+00:00"
    );
}

#[tokio::test]
async fn limit_order_sends_the_price_without_market_protection() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string_contains("order_type=LIMIT"))
        .and(body_string_contains("price=1550.5"))
        .respond_with(ok(json!({"order_id": "250101000000001"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/orders/250101000000001"))
        .respond_with(ok(json!([order_record(
            "OPEN",
            json!({"order_type": "LIMIT", "price": 1550.5})
        )])))
        .mount(&server)
        .await;
    let request = OrderRequest {
        order_type: OrderType::Limit,
        limit_price: Some(1550.5),
        ..buy(5.0)
    };

    let order = backend(&server, true)
        .await
        .place_order(&request)
        .await
        .unwrap();

    assert_eq!(order.status, OrderStatus::Submitted);
    let sent =
        String::from_utf8(server.received_requests().await.unwrap()[0].body.clone()).unwrap();
    assert!(!sent.contains("market_protection"), "{sent}");
}

#[tokio::test]
async fn rejected_order_carries_kites_reason() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok(json!({"order_id": "250101000000001"})))
        .mount(&server)
        .await;
    Mock::given(path("/orders/250101000000001"))
        .respond_with(ok(json!([order_record(
            "REJECTED",
            json!({"status_message": "Insufficient funds. Required margin is 95000"})
        )])))
        .mount(&server)
        .await;

    let err = backend(&server, true)
        .await
        .place_order(&buy(5.0))
        .await
        .unwrap_err();

    assert!(matches!(err, ServiceError::OrderRejected(m) if m.contains("Insufficient funds")));
}

#[tokio::test]
async fn unregistered_ip_refusal_explains_the_static_ip_rule() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(kite_error(403, "PermissionException", "IP not allowed"))
        .mount(&server)
        .await;

    let err = backend(&server, true)
        .await
        .place_order(&buy(5.0))
        .await
        .unwrap_err();

    assert!(matches!(err, ServiceError::OrderRejected(m) if m.contains("registered static IP")));
}

#[tokio::test]
async fn fractional_order_below_one_share_is_never_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let err = backend(&server, true)
        .await
        .place_order(&buy(0.9))
        .await
        .unwrap_err();

    assert!(matches!(err, ServiceError::OrderRejected(m) if m.contains("rounds down to 0")));
}

#[tokio::test]
async fn balance_and_positions_merge_holdings_with_todays_positions() {
    let server = MockServer::start().await;
    Mock::given(path("/user/margins/equity"))
        .respond_with(ok(
            json!({"net": 50000.0, "available": {"live_balance": 45000.0}}),
        ))
        .mount(&server)
        .await;
    Mock::given(path("/portfolio/holdings"))
        .respond_with(ok(
            json!([{"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 10,
                                 "t1_quantity": 2, "average_price": 1500.0, "last_price": 1600.0}]),
        ))
        .mount(&server)
        .await;
    Mock::given(path("/portfolio/positions"))
        .respond_with(ok(json!({"net": [{"tradingsymbol": "INFY", "exchange": "NSE", "product": "CNC",
                                         "quantity": 3, "average_price": 1580.0, "last_price": 1600.0}]})))
        .mount(&server)
        .await;
    let kite = backend(&server, false).await;

    let balance = kite.balance().await.unwrap();

    assert_eq!(balance.cash, 45000.0);
    assert_eq!(balance.buying_power, 50000.0);
    assert_eq!(balance.equity, 45000.0 + 15.0 * 1600.0);
}

#[tokio::test]
async fn historical_bars_are_chunked_and_the_instrument_dump_is_cached() {
    let server = MockServer::start().await;
    let dump = "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange\n\
408065,1594,INFY,INFOSYS,0,,0,0.05,1,EQ,NSE,NSE\n";
    let gzipped = {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(dump.as_bytes()).unwrap();
        encoder.finish().unwrap()
    };
    Mock::given(path("/instruments/NSE"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(gzipped))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/instruments/historical/408065/day"))
        .and(query_param("from", "2018-01-01 00:00:00"))
        .and(query_param("to", "2023-06-23 23:59:59"))
        .respond_with(ok(
            json!({"candles": [["2020-01-02T00:00:00+0530", 1, 2, 0.5, 1.5, 100]]}),
        ))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/instruments/historical/408065/day"))
        .and(query_param("from", "2023-06-24 00:00:00"))
        .respond_with(ok(
            json!({"candles": [["2024-11-01T00:00:00+0530", 1, 2, 0.5, 1.6, 100]]}),
        ))
        .expect(2)
        .mount(&server)
        .await;
    let kite = backend(&server, false).await;
    let (start, end) = (
        NaiveDate::from_ymd_opt(2018, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2024, 11, 3).unwrap(),
    );

    let bars = kite
        .historical_bars("INFY", start, end, Timeframe::Day)
        .await
        .unwrap();
    kite.historical_bars(
        "INFY",
        NaiveDate::from_ymd_opt(2023, 6, 24).unwrap(),
        end,
        Timeframe::Day,
    )
    .await
    .unwrap();

    assert_eq!(
        bars.iter()
            .map(|b| b.timestamp.as_str())
            .collect::<Vec<_>>(),
        ["2020-01-02T00:00:00", "2024-11-01T00:00:00"]
    );
}

#[tokio::test]
async fn cancel_only_touches_open_orders() {
    let server = MockServer::start().await;
    Mock::given(path("/orders/1"))
        .respond_with(ok(json!([order_record("OPEN", json!({"order_id": "1"}))])))
        .mount(&server)
        .await;
    Mock::given(path("/orders/2"))
        .respond_with(ok(json!([order_record(
            "COMPLETE",
            json!({"order_id": "2", "filled_quantity": 9})
        )])))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/orders/regular/1"))
        .respond_with(ok(json!({"order_id": "1"})))
        .expect(1)
        .mount(&server)
        .await;
    let kite = backend(&server, true).await;

    kite.cancel_order("1").await.unwrap();
    kite.cancel_order("2").await.unwrap();
}

#[tokio::test]
async fn an_accepted_order_keeps_its_id_when_the_status_check_fails() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok(json!({"order_id": "250101000000001"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/orders/250101000000001"))
        .respond_with(kite_error(429, "NetworkException", "Too many requests"))
        .mount(&server)
        .await;

    let order = backend(&server, true)
        .await
        .place_order(&buy(9.7))
        .await
        .expect("an accepted order must not be reported as an error");

    assert_eq!(order.order_id, "250101000000001");
    assert_eq!(order.status, OrderStatus::Pending);
    assert_eq!(order.quantity, 9.0);
}
