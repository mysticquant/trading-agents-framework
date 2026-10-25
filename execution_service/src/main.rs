use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use tokio::net::TcpListener;

use execd::backend::Backend;
use execd::kite::{KiteBackend, KiteConfig};
use execd::paper::PaperBackend;
use execd::server;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    /// Simulated ledger with immediate fills (no real broker).
    Paper,
    /// Zerodha Kite Connect (real money; orders need --allow-live).
    Kite,
}

/// Local order-execution service for trading-agents.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Loopback address to listen on (port 0 picks a free port).
    #[arg(long, default_value = "127.0.0.1:7600")]
    listen: SocketAddr,

    #[arg(long, value_enum, default_value = "paper")]
    backend: BackendKind,

    /// Allow real-money orders on the kite backend. Without it, Kite orders
    /// are refused and only reads work.
    #[arg(long)]
    allow_live: bool,

    /// Shared secret every request must carry.
    #[arg(long, env = "EXECD_TOKEN", hide_env_values = true)]
    token: String,

    /// Paper backend: starting cash.
    #[arg(long, default_value_t = 100_000.0)]
    initial_cash: f64,
    /// Paper backend: commission in basis points.
    #[arg(long, default_value_t = 5.0)]
    commission_bps: f64,
    /// Paper backend: slippage in basis points.
    #[arg(long, default_value_t = 5.0)]
    slippage_bps: f64,

    #[arg(long, env = "KITE_API_KEY", default_value = "")]
    kite_api_key: String,
    #[arg(
        long,
        env = "KITE_ACCESS_TOKEN",
        default_value = "",
        hide_env_values = true
    )]
    kite_access_token: String,
    /// Kite API base URL (override only for testing).
    #[arg(long, env = "KITE_BASE_URL", default_value = execd::kite::API_BASE)]
    kite_base_url: String,
    #[arg(long, default_value = "NSE")]
    exchange: String,
    #[arg(long, default_value = "CNC")]
    product: String,
}

#[tokio::main]
async fn main() {
    // Logs go to stderr; stdout carries only the readiness line below.
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
    if let Err(message) = run(Args::parse()).await {
        eprintln!("execd: {message}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<(), String> {
    // The service sends orders on request; never expose that to the network.
    if !args.listen.ip().is_loopback() {
        return Err(format!(
            "refusing to listen on non-loopback address {}",
            args.listen
        ));
    }
    if args.token.len() < 16 {
        return Err("EXECD_TOKEN must be at least 16 characters".into());
    }

    let backend = match args.backend {
        BackendKind::Paper => Backend::Paper(PaperBackend::new(
            args.initial_cash,
            args.commission_bps,
            args.slippage_bps,
        )),
        BackendKind::Kite => {
            if args.kite_api_key.is_empty() {
                return Err("KITE_API_KEY is required for the kite backend".into());
            }
            let mut config = KiteConfig::new(args.kite_api_key, args.kite_access_token);
            config.base_url = args.kite_base_url;
            config.exchange = args.exchange;
            config.product = args.product;
            config.allow_live = args.allow_live;
            config.ack_timeout = Duration::from_secs(10);
            Backend::Kite(Box::new(
                KiteBackend::new(config).map_err(|e| e.to_string())?,
            ))
        }
    };
    backend.connect().await.map_err(|e| e.to_string())?;

    let listener = TcpListener::bind(args.listen)
        .await
        .map_err(|e| format!("cannot listen on {}: {e}", args.listen))?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    // Machine-readable readiness line (the port matters with --listen :0).
    println!(
        "execd listening on {address} backend={} orders_enabled={}",
        backend.name(),
        backend.orders_enabled()
    );
    if matches!(backend, Backend::Kite(_)) && backend.orders_enabled() {
        tracing::warn!("LIVE: kite orders are enabled and will use real money");
    }

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    server::serve(listener, Arc::new(backend), Arc::new(args.token), shutdown).await;
    Ok(())
}
