//! Pins the invariant #94 established and this file's own doc comments
//! document: on native, `RerankClient` contributes no bound of its own — the
//! call-site `tokio::time::timeout` (in `alaya-core::service`) is the only
//! thing that can end a stalled call. If a client-side timer is ever
//! reintroduced here, this test starts passing for the wrong reason (the
//! outer timeout stops racing the client's) rather than failing loudly, so
//! it asserts the client alone never resolves within a budget multiple.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Duration;

use alaya_backends::RerankingService;
use alaya_backends::rerank::RerankClient;
use tokio::net::TcpListener;

#[tokio::test]
async fn native_rerank_client_never_resolves_on_its_own() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Accept every connection and hold it open without ever writing a
    // response, so the client is left waiting for a reply that never comes.
    // The socket is leaked (not dropped) so it doesn't reset the connection.
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            std::mem::forget(socket);
        }
    });

    let budget = Duration::from_millis(200);
    let client = RerankClient::new(format!("http://{addr}"), 5, None, budget)
        .expect("client builds against a local address");

    // If `RerankClient` carried its own transport timer at or below
    // `budget`, this would resolve well inside `outer` instead of racing
    // past it.
    let outer = budget * 3;
    let result = tokio::time::timeout(outer, client.rerank("query", &["doc"])).await;

    assert!(
        result.is_err(),
        "RerankClient::rerank resolved on its own before {outer:?} elapsed — \
         the native client must not enforce a transport timeout; the caller's \
         tokio::time::timeout is the only bound"
    );
}
