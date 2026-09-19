//! One reqwest builder for every upstream client, and the one bounded body
//! read they all go through. Redirects are refused everywhere: a 3xx must
//! never be able to carry a server-held credential off-host.

use std::time::Duration;

/// Ceiling on any upstream response body. reqwest reads to EOF with no
/// default cap, so a compromised or MITM'd upstream answering with a
/// multi-gigabyte — or endless — body would grow the heap until the OOM
/// killer takes the pod (limit 256Mi, `deploy/console/ops-console.yaml`).
/// The request timeout is no defence: it bounds the seconds, not the bytes a
/// fast in-cluster link delivers inside them.
///
/// This is a DoS bound, not a promise that no honest reply is ever refused.
/// A search page is `PAGE_SIZE` memories and alaya-server accepts 1 MiB
/// request bodies, so twenty maximal memories could structurally exceed it —
/// but that needs ~400 KB averaged across every row, and the refusal is a
/// named 502 rather than a pod restart. Sizing to that structural ceiling
/// instead would not fit 256Mi: `serde_json::Value` expands the text
/// several-fold, for two bodies at once on the pages that `join!` a pair.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// No `Debug`, deliberately: a `reqwest::Error` can embed the full request
/// URL, which may carry userinfo. `AppError::body` is the only exit, and it
/// renders through `AppError::transport`, which keeps one phrase.
pub enum BodyError {
    Transport(reqwest::Error),
    /// The upstream ran past `MAX_BODY_BYTES`; the read was abandoned there.
    TooLarge,
}

pub fn client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build http client")
}

/// Read a response body as text, refusing anything over `MAX_BODY_BYTES`.
/// Overrun drops the connection mid-body, so nothing past the cap is ever
/// buffered. Lossy UTF-8 matches what `Response::text()` does without the
/// `charset` feature, which is how this workspace builds reqwest.
///
/// `what` names the upstream for the log line — a body that trips the cap is
/// an upstream misbehaving, and some callers (the per-card degrade in
/// `routes/home.rs`) drop the error, so without this it is invisible.
pub async fn body_text(what: &str, mut resp: reqwest::Response) -> Result<String, BodyError> {
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(BodyError::Transport)? {
        if buf.len() + chunk.len() > MAX_BODY_BYTES {
            tracing::warn!(
                upstream = what,
                cap = MAX_BODY_BYTES,
                "upstream response body exceeded the cap — read abandoned"
            );
            return Err(BodyError::TooLarge);
        }
        buf.extend_from_slice(&chunk);
    }
    // Consume `buf` on the valid path: a borrow-then-`into_owned` would hold
    // both copies live and double the peak the cap is there to bound.
    Ok(String::from_utf8(buf)
        .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve `len` bytes on loopback and read them back through `body_text`.
    async fn read_body_of(len: usize) -> Result<String, BodyError> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || async move { "x".repeat(len) }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let resp = client(Duration::from_secs(10))
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap();
        let out = body_text("test", resp).await;
        server.abort();
        out
    }

    #[tokio::test]
    async fn oversized_upstream_body_is_refused_not_buffered() {
        // Over loopback the body arrives in many chunks, so this exercises
        // the running total — not just a single oversized first chunk.
        assert!(matches!(
            read_body_of(MAX_BODY_BYTES + 1024).await,
            Err(BodyError::TooLarge)
        ));
        // A body inside the cap still reads in full: this is not truncation.
        assert!(matches!(read_body_of(64 * 1024).await, Ok(s) if s.len() == 64 * 1024));
    }
}
