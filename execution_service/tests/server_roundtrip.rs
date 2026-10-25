//! The real TCP server with the paper backend, driven over a socket.

use std::sync::Arc;

use execd::backend::Backend;
use execd::paper::PaperBackend;
use execd::server;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const TOKEN: &str = "test-token-0123456789";

struct Client {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    writer: tokio::net::tcp::OwnedWriteHalf,
    next_id: u64,
}

impl Client {
    async fn call(&mut self, method: &str, params: Value) -> Value {
        self.call_with_token(method, params, TOKEN).await
    }

    async fn call_with_token(&mut self, method: &str, params: Value, token: &str) -> Value {
        self.next_id += 1;
        let request =
            json!({"id": self.next_id, "token": token, "method": method, "params": params});
        self.writer
            .write_all(format!("{request}\n").as_bytes())
            .await
            .unwrap();
        let mut line = String::new();
        self.reader.read_line(&mut line).await.unwrap();
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], self.next_id, "responses match requests");
        response
    }
}

async fn start() -> Client {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let backend = Backend::Paper(PaperBackend::new(10_000.0, 0.0, 0.0));
    tokio::spawn(server::serve(
        listener,
        Arc::new(backend),
        Arc::new(TOKEN.to_string()),
        std::future::pending(),
    ));
    let (read, write) = TcpStream::connect(address).await.unwrap().into_split();
    Client {
        reader: BufReader::new(read),
        writer: write,
        next_id: 0,
    }
}

#[tokio::test]
async fn full_order_lifecycle_over_the_socket() {
    let mut client = start().await;

    let health = client.call("health", Value::Null).await;
    assert_eq!(health["result"]["backend"], "paper");
    assert_eq!(health["result"]["orders_enabled"], true);

    client
        .call("mark_price", json!({"ticker": "AAPL", "price": 100.0}))
        .await;
    let order = client
        .call(
            "place_order",
            json!({"ticker": "AAPL", "side": "buy", "quantity": 12.5, "order_type": "market"}),
        )
        .await;
    assert_eq!(order["result"]["status"], "filled");
    assert_eq!(order["result"]["quantity"], 12.0);
    let order_id = order["result"]["order_id"].as_str().unwrap().to_string();

    let positions = client.call("get_positions", Value::Null).await;
    assert_eq!(positions["result"]["AAPL"]["quantity"], 12.0);
    let balance = client.call("get_account_balance", Value::Null).await;
    assert_eq!(balance["result"]["cash"], 10_000.0 - 1_200.0);

    let status = client
        .call("get_order_status", json!({"order_id": order_id}))
        .await;
    assert_eq!(status["result"]["order_id"], order_id);
}

#[tokio::test]
async fn errors_come_back_typed_and_the_connection_stays_usable() {
    let mut client = start().await;

    let unauthorized = client.call_with_token("health", Value::Null, "wrong").await;
    assert_eq!(unauthorized["error"]["type"], "Unauthorized");

    let rejected = client
        .call(
            "place_order",
            json!({"ticker": "AAPL", "side": "sell", "quantity": 1, "order_type": "market"}),
        )
        .await;
    assert_eq!(rejected["error"]["type"], "OrderRejected");

    let bad_params = client.call("place_order", json!({"ticker": "AAPL"})).await;
    assert_eq!(bad_params["error"]["type"], "InvalidRequest");

    let unsupported = client
        .call(
            "get_historical_bars",
            json!({"ticker": "AAPL", "start": "2024-01-01", "end": "2024-01-31", "timeframe": "1d"}),
        )
        .await;
    assert_eq!(unsupported["error"]["type"], "NotSupported");

    // Still serving after all of that.
    assert_eq!(
        client.call("health", Value::Null).await["result"]["backend"],
        "paper"
    );
}
