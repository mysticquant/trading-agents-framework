//! Errors carried back to the client. Each kind maps onto one Python
//! exception in the proxy adapter: OrderRejected -> OrderRejectedError,
//! AuthExpired -> BrokerAuthExpiredError, everything else -> BrokerError.

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ServiceError {
    /// The broker (or a safety rule) refused this order.
    #[error("{0}")]
    OrderRejected(String),
    /// The broker session expired; the operator must log in again.
    #[error("{0}")]
    AuthExpired(String),
    /// Missing or wrong service token.
    #[error("{0}")]
    Unauthorized(String),
    /// Malformed request (bad JSON, unknown method, bad params).
    #[error("{0}")]
    InvalidRequest(String),
    /// The backend doesn't implement this method.
    #[error("{0}")]
    NotSupported(String),
    /// Any other broker or network failure.
    #[error("{0}")]
    Broker(String),
}

impl ServiceError {
    pub fn kind(&self) -> &'static str {
        match self {
            ServiceError::OrderRejected(_) => "OrderRejected",
            ServiceError::AuthExpired(_) => "AuthExpired",
            ServiceError::Unauthorized(_) => "Unauthorized",
            ServiceError::InvalidRequest(_) => "InvalidRequest",
            ServiceError::NotSupported(_) => "NotSupported",
            ServiceError::Broker(_) => "Broker",
        }
    }
}
