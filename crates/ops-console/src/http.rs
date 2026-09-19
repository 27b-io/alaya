//! One reqwest builder for every upstream client. Redirects are refused
//! everywhere: a 3xx must never be able to carry a server-held credential
//! off-host.

use std::time::Duration;

pub fn client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build http client")
}
