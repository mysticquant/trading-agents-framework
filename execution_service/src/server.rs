//! TCP server: newline-delimited JSON requests, one task per connection,
//! requests on a connection handled in order. Every request must carry the
//! shared service token, compared in constant time.

use std::future::Future;
use std::io;
use std::sync::Arc;

use subtle::ConstantTimeEq;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::backend::Backend;
use crate::error::ServiceError;
use crate::protocol::{Request, Response};

/// Requests larger than this are refused (and the connection closed) so a
/// misbehaving client can't make the service buffer without bound.
pub const MAX_REQUEST_BYTES: usize = 64 * 1024;

pub async fn serve(
    listener: TcpListener,
    backend: Arc<Backend>,
    token: Arc<String>,
    shutdown: impl Future<Output = ()>,
) {
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, peer)) => {
                    tracing::debug!(%peer, "client connected");
                    tokio::spawn(handle_connection(stream, backend.clone(), token.clone()));
                }
                Err(err) => tracing::warn!(error = %err, "accept failed"),
            },
            () = &mut shutdown => {
                tracing::info!("shutting down");
                return;
            }
        }
    }
}

async fn handle_connection(stream: TcpStream, backend: Arc<Backend>, token: Arc<String>) {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    loop {
        let response = match read_bounded_line(&mut reader, MAX_REQUEST_BYTES).await {
            Ok(None) => return, // client closed the connection
            Ok(Some(line)) if line.trim().is_empty() => continue,
            Ok(Some(line)) => process(&line, &backend, &token).await,
            Err(err) => {
                let response = Response::from_result(
                    0,
                    Err(ServiceError::InvalidRequest(format!(
                        "unreadable request: {err}"
                    ))),
                );
                let _ = write_response(&mut write_half, &response).await;
                return;
            }
        };
        if write_response(&mut write_half, &response).await.is_err() {
            return;
        }
    }
}

async fn write_response(
    writer: &mut (impl AsyncWriteExt + Unpin),
    response: &Response,
) -> io::Result<()> {
    let mut line = serde_json::to_vec(response).expect("responses always serialize");
    line.push(b'\n');
    writer.write_all(&line).await?;
    writer.flush().await
}

pub async fn process(line: &str, backend: &Backend, token: &str) -> Response {
    let request: Request = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(err) => {
            return Response::from_result(
                0,
                Err(ServiceError::InvalidRequest(format!(
                    "invalid JSON request: {err}"
                ))),
            );
        }
    };
    if !bool::from(request.token.as_bytes().ct_eq(token.as_bytes())) {
        return Response::from_result(
            request.id,
            Err(ServiceError::Unauthorized(
                "missing or wrong service token".into(),
            )),
        );
    }
    Response::from_result(
        request.id,
        backend.handle(&request.method, request.params).await,
    )
}

/// Read one '\n'-terminated line without buffering more than `max` bytes.
/// Returns `None` at a clean end of stream.
async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> io::Result<Option<String>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok((!line.is_empty()).then(|| String::from_utf8_lossy(&line).into_owned()));
        }
        let (chunk, done) = match available.iter().position(|&b| b == b'\n') {
            Some(end) => (&available[..end], Some(end + 1)),
            None => (available, None),
        };
        if line.len() + chunk.len() > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request too large",
            ));
        }
        line.extend_from_slice(chunk);
        let consumed = done.unwrap_or(chunk.len());
        reader.consume(consumed);
        if done.is_some() {
            return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::PaperBackend;

    fn paper() -> Backend {
        Backend::Paper(PaperBackend::new(1000.0, 0.0, 0.0))
    }

    #[tokio::test]
    async fn rejects_a_wrong_token() {
        let response = process(
            r#"{"id": 3, "token": "nope", "method": "health"}"#,
            &paper(),
            "secret",
        )
        .await;
        assert_eq!(response.id, 3);
        assert_eq!(response.error.unwrap().kind, "Unauthorized");
    }

    #[tokio::test]
    async fn rejects_malformed_json_and_unknown_methods() {
        let bad = process("{not json", &paper(), "secret").await;
        assert_eq!(bad.error.unwrap().kind, "InvalidRequest");
        let unknown = process(
            r#"{"id": 1, "token": "secret", "method": "nope"}"#,
            &paper(),
            "secret",
        )
        .await;
        assert!(unknown.error.unwrap().message.contains("unknown method"));
    }

    #[tokio::test]
    async fn bounded_reader_splits_lines_and_refuses_huge_ones() {
        let mut reader = BufReader::new(&b"first\nsecond\n"[..]);
        assert_eq!(
            read_bounded_line(&mut reader, 100)
                .await
                .unwrap()
                .as_deref(),
            Some("first")
        );
        assert_eq!(
            read_bounded_line(&mut reader, 100)
                .await
                .unwrap()
                .as_deref(),
            Some("second")
        );
        assert_eq!(read_bounded_line(&mut reader, 100).await.unwrap(), None);

        let huge = vec![b'x'; 200];
        let mut reader = BufReader::new(huge.as_slice());
        assert!(read_bounded_line(&mut reader, 100).await.is_err());
    }
}
