//! One reqwest builder for every upstream client, and the one bounded body
//! read they all go through. Redirects are refused everywhere: a 3xx must
//! never be able to carry a server-held credential off-host. Proxy env vars
//! are ignored for the same reason: the peer dialled must be the host
//! `validate_upstream_url` classified, not whatever `HTTP_PROXY` names.

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
        // Dial the host `validate_upstream_url` classified, never an env proxy.
        .no_proxy()
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
    while let Some(chunk) = resp.chunk().await.map_err(|e| {
        // `AppError::transport` only builds a string and the per-card
        // degrade in `routes/home.rs` drops the error outright, so without
        // this a body that dies mid-stream is recorded nowhere.
        tracing::warn!(
            upstream = what,
            timeout = e.is_timeout(),
            "upstream body read failed mid-stream"
        );
        BodyError::Transport(e)
    })? {
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

    /// Serve `app` on an ephemeral loopback port (`:0` — the kernel picks,
    /// so nothing here is port-dependent). Abort the handle when done.
    async fn serve(app: axum::Router) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            // Unreachable, not discarded. `axum::serve` is typed to yield
            // `io::Result<()>` but documents that it never completes or
            // returns an error: accept errors are handled inside its own
            // loop (log, sleep one second, retry), and no graceful-shutdown
            // signal is wired here, so the future only ever ends by being
            // dropped at `handle.abort()`. There is no error value to
            // surface. The failure that IS observable — the bind — is the
            // `unwrap()` above, which is where a real one shows up.
            let _ = axum::serve(listener, app).await;
        });
        (addr, handle)
    }

    /// Serve `len` bytes on loopback and read them back through `body_text`.
    async fn read_body_of(len: usize) -> Result<String, BodyError> {
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || async move { "x".repeat(len) }),
        );
        let (addr, server) = serve(app).await;
        let resp = client(Duration::from_secs(10))
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap();
        let out = body_text("test", resp).await;
        server.abort();
        out
    }

    /// The SSRF guard the three OIDC fetches lean on: `client()` must refuse
    /// to follow a 3xx. reqwest already strips `Authorization` on a
    /// cross-host hop, so the exposure is not header replay — it is that a
    /// 307/308 replays the REQUEST BODY, and the token POST's body carries
    /// the authorization `code` and the PKCE `code_verifier`, a one-shot
    /// credential redeemable for an id_token. A hostile or compromised IdP
    /// answering the token endpoint with a redirect would receive both.
    ///
    /// It also keeps every fetch on the origin `same_origin_https` just
    /// validated; a followed 3xx would move it off one.
    ///
    /// `client()` is the only `Client::builder()` in the crate, so this one
    /// assertion covers all four upstreams. Nothing else pins the policy.
    #[tokio::test]
    async fn client_refuses_to_follow_redirects() {
        let app = axum::Router::new()
            .route(
                "/",
                axum::routing::get(|| async { axum::response::Redirect::temporary("/followed") }),
            )
            .route("/followed", axum::routing::get(|| async { "FOLLOWED" }));
        let (addr, server) = serve(app).await;
        let resp = client(Duration::from_secs(10))
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap();
        server.abort();
        // Following it would return 200 from /followed instead.
        assert_eq!(resp.status().as_u16(), 307);
    }

    /// The peer dialled must be the host `validate_upstream_url` classified,
    /// so `client()` ignores `HTTP_PROXY` and friends (LAB-4695). reqwest
    /// reads them at `build()` and `set_var` races this binary, so the probe
    /// runs in a child that inherits them: a trap listener named as the proxy
    /// must see no connection when `client()` dials a refused loopback port.
    /// A default client must see one, or the harness proves nothing.
    #[test]
    fn client_ignores_system_proxy() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const CHILD: &str = "OPS_CONSOLE_PROXY_TEST_CHILD";
        const TARGET: &str = "http://127.0.0.1:1";
        const PROBED: &str = "proxy probe complete";
        if std::env::var_os(CHILD).is_none() {
            let trap = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let proxy = format!("http://{}", trap.local_addr().unwrap());
            drop(trap); // the child binds it again; only the port is needed
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "http::tests::client_ignores_system_proxy",
                    "--exact",
                    "--nocapture",
                ])
                .env(CHILD, &proxy)
                .env("HTTP_PROXY", &proxy)
                .env("HTTPS_PROXY", &proxy)
                .env("ALL_PROXY", &proxy)
                .env_remove("NO_PROXY")
                .env_remove("no_proxy")
                // A set REQUEST_METHOD makes reqwest ignore HTTP_PROXY (CGI).
                .env_remove("REQUEST_METHOD")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            // libtest exits 0 when `--exact` matches nothing, so success alone
            // passes a renamed or moved test with no client probed.
            assert!(
                String::from_utf8_lossy(&out.stdout).contains(PROBED),
                "child never reached the end of the probe"
            );
            return;
        }

        let proxy = std::env::var(CHILD).unwrap();
        let trap = std::net::TcpListener::bind(proxy.trim_start_matches("http://")).unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        std::thread::spawn(move || {
            for stream in trap.incoming() {
                counter.fetch_add(1, Ordering::SeqCst);
                drop(stream); // the client sees a reset and returns
            }
        });

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let _ = reqwest::Client::new().get(TARGET).send().await;
        });
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "harness broken: a default client did not use the env proxy"
        );
        rt.block_on(async {
            let _ = client(Duration::from_secs(5)).get(TARGET).send().await;
        });
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "client() dialled through the env proxy"
        );
        println!("{PROBED}");
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
