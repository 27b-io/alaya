# ops-console

OIDC-gated web console for the 27b workspace (LAB-1684 / LAB-1641). Two
route modules in one crate: Ālaya memory curation, and the anthropic-lb
read-only monitoring pane (LAB-1964).

## Trust model (D2 — ratified 2026-08-15, do not re-litigate here)

The console backend is a **trusted service consumer** of alaya-server, the
same class as radar and unified-memory:

- The **browser** holds only an encrypted session cookie, minted after an
  OIDC authorization-code login (PKCE S256, nonce, pinned `redirect_uri`)
  against id.27b.io, gated by a **default-deny subject allowlist**
  (`CONSOLE_ALLOWED_SUBJECTS`). A non-allowlisted subject gets an explicit
  403 — never a degraded session.
- Every alaya-server call executes **server-side with the static bearer**
  (`ALAYA_API_KEY`). No token, bearer, or client secret ever reaches the
  browser — there is no JS bundle at all (SSR-only Leptos, plain HTML forms).
- Ālaya's own OIDC principals stay **read / additive** (store only, never
  delete / supersede / merge / relation / patch / backfill); this console adds zero new
  authorization semantics to alaya-server and does not pre-empt the LAB-1084
  ACL/namespace decision.

Session posture (LAB-1694 pass/fail set + team session rules): CSRF token on
every form + strict `Origin` check on every state-changing request (anything
other than `GET`/`HEAD`), `HttpOnly`/`SameSite=Lax`/`Secure` cookies, a fresh
session cookie minted on the OIDC callback (fixation defense), 12 h absolute
session lifetime **and** 15 min idle timeout (sliding — refreshed on every
authenticated request), server-side logout revocation (a logged-out session
id is rejected until its absolute expiry, in-memory, single-replica), strict
CSP (`default-src 'none'`).

## Configuration (fail-closed — missing anything below refuses startup)

| Variable | Meaning |
|:---------|:--------|
| `CONSOLE_PUBLIC_URL` | External base URL — **must be https** (plain http is accepted only for loopback local dev; anything else refuses startup). `redirect_uri` is pinned to `{url}/auth/callback` — register exactly that on the IdP client. |
| `CONSOLE_OIDC_ISSUER` | `https://id.27b.io` |
| `CONSOLE_OIDC_CLIENT_ID` / `CONSOLE_OIDC_CLIENT_SECRET` | Confidential OIDC client (authorization-code + PKCE). |
| `CONSOLE_ALLOWED_SUBJECTS` | Comma-separated OIDC `sub` values allowed to log in. Default-deny; must be non-empty. |
| `CONSOLE_SESSION_SECRET` | ≥ 32 bytes; expanded (SHA-512) into the cookie-encryption key. Rotating it invalidates all sessions. |
| `ALAYA_URL` | `http://alaya-server.mcp.svc:3001` |
| `ALAYA_API_KEY` | Static bearer (full write) — server-side only. |
| `CONSOLE_LISTEN_ADDR` | Optional, default `0.0.0.0:3002`. |
| `LB_URL` / `LB_API_KEY` / `METRICS_URL` | anthropic-lb module — **all three or none**. None: the module is disabled and the home card says so. A partial set refuses startup. `LB_URL` = the LB's base URL; `LB_API_KEY` = an LB **operator** client key, sent server-side as `x-api-key` (the LB rejects `Authorization: Bearer` on `/_stats`); `METRICS_URL` = a Prometheus-compatible query API for the 7-day history — the console only ever calls `/api/v1/query_range` on it, so point it at a route that exposes nothing else. |

Every upstream URL (`ALAYA_URL`, `LB_URL`, `METRICS_URL`) must be https, or plain http to a cluster-local host (`*.svc`, `*.svc.cluster.local`, `*.internal`, a single-label name, or a loopback/private IP literal), and must carry no query string or fragment. Anything else refuses startup — `METRICS_URL` included, keyless or not: off-cluster plaintext leaves the budget history both readable and rewritable in flight. `CONSOLE_OIDC_ISSUER` is stricter still: https only, no cluster-local exemption, and no userinfo (it is printed verbatim in the startup config log).

Logs are JSON, one object per line on stdout, so structured fields (`sub`,
`op`, `issuer`, `cause`, ...) are queryable as `fields.<name>` rather than
scraped from text. `RUST_LOG` defaults to `ops_console=info,tower_http=info`.

### Allowlisting an admin

1. Have them log in once (they'll get a 403 page); the rejected subject is
   the `fields.sub` value on the console's `login rejected: subject not
   allowlisted` record. (Or read the `sub` from the IdP's user admin.)
2. Add it to `CONSOLE_ALLOWED_SUBJECTS` (comma-separated) and roll the pod.

## Ālaya module

- **Browse/search** — hybrid / scan / recent / tag modes, type + tag filters,
  superseded-visibility toggle, pagination.
- **Detail** — full content, metadata, salience/access/trust stats,
  relations, supersession chain (audit trail rendered, never hidden).
- **Curation** — supersede (reason required), correct-&-supersede (store a
  fixed copy, then supersede the original), delete (two-step confirm),
  merge duplicates (dry-run preview before commit), relations
  create/delete, contradiction resolution via supersede.
- **Auth state** — read-only view of alaya-server's `GET /auth/config`:
  principal × operation matrix + OIDC issuer/audience/allowlist.

## anthropic-lb module (read-only)

Scope: the console **renders** LB state; budgets, limits, endpoints and
client identities change through GitOps only. The console has **no write
route to the LB**, the LB exposes no admin write API, and this module must
not grow one. Values render with their provenance; limits are
"TOML, GitOps".

- **Fleet** — routing strategy, replicas seen, shared-state (Redis) health,
  pooled headroom, cumulative upstream transport errors. Source:
  `GET /_stats`. Only fleet-wide values render: `/_stats` also carries
  process-local counters (per-consumer request rates, per-endpoint burn
  rates) that describe one random replica behind a Service, so they are
  deliberately left out.
- **Per-client budget burn** — today's fleet-wide used / limit with a
  progress bar (`/_stats` → `cluster.budget_usage`, the Redis aggregate).
  When that aggregate is absent or empty the card falls back to the
  replica-local `client_budgets` mirror and says so with a "replica-local"
  badge — that mirror resets on pod restart and undercounts the fleet. Plus
  a 7-column history: the daily peak of `anthropic_cluster_budget_used` per
  UTC day, today's column running. History is one `query_range` against
  `METRICS_URL` (`max by (client) (max_over_time(…[23h58m]))` at 23:59 UTC
  of each day — the trimmed window keeps the first post-midnight scrape,
  which can still carry yesterday's total, out of today's peak). No history
  store in the console, no dashboard embeds.
- **Upstream accounts** — per endpoint: 5-hour / 7-day window utilisation,
  hard-limit / throttle status, remaining requests, next 5h reset; hottest
  first.

Each card degrades on its own: a dark metrics store leaves live headroom
up, an LB outage leaves the burn history up. Every upstream client in the
console refuses redirects (no credential ever rides a 3xx off-host).

## Deploy

Deployed from the private infra repo's Kubernetes manifests (LAB-2712).
`deploy/console/ops-console.yaml` here is the runtime-contract template
(Deployment + Service: image, command, env, probes, security context). The
deployed manifest — including the NetworkPolicy / egress allowlist, real
hostnames and the digest pin — lives in the private infra repo and is the
truth; cluster topology is not published from this repository. Binding
shape wherever this binary runs: own label, own NetworkPolicy with egress
pinned to the module upstreams + IdP :443 only, **no dragonfly egress**,
image digest-pinned via the `flux-system:alaya` imagepolicy marker so the
console rolls with alaya-server. The binary ships in the existing public
`ghcr.io/27b-io/alaya` image (`command: ["ops-console"]`), pulled anonymously
since LAB-3719 — no pull secret.

Config split: `CONSOLE_PUBLIC_URL`, `CONSOLE_OIDC_ISSUER`, `ALAYA_URL`,
`LB_URL`, `METRICS_URL` are plain env in the manifest;
`CONSOLE_OIDC_CLIENT_ID`, `CONSOLE_OIDC_CLIENT_SECRET`,
`CONSOLE_ALLOWED_SUBJECTS`, `CONSOLE_SESSION_SECRET`, `ALAYA_API_KEY` and
`LB_API_KEY` come from a secret manager, rendered by ESO into Secret
`ops-console-env`. Editing the Secret rolls the pod (Reloader annotation).
`LB_URL`, `METRICS_URL` (manifest) and `LB_API_KEY` (Secret) are one
all-or-nothing group: land all three in the same change — a half-set group
refuses startup by design (see the `LB_URL` comment in
`deploy/console/ops-console.yaml`).

Tailnet HTTPS — order matters: define `svc:ops` in the Tailscale admin
console **first**, then on the lab node run `tailscale serve --bg --service
svc:ops --https 443 http://ops-console.mcp.svc.cluster.local:3002`.
CLI-first exits 0 but the service never appears in the console for approval.

Post-rollout egress check (the image has `curl`, not `nc`; k3s netpol rejects,
so expect "Connection refused" on the first two and `alaya=200`):

```bash
kubectl -n mcp exec deploy/ops-console -- sh -c '
  curl -sS -m3 telnet://dragonfly.mcp.svc:6379; echo dragonfly_exit=$?;
  curl -sS -m3 telnet://alaya-bridge.mcp.svc:3000; echo bridge_exit=$?;
  curl -sS -m3 -o /dev/null -w "alaya=%{http_code}\n" http://alaya-server.mcp.svc:3001/health'
```

The equivalent probes for the LB pane's two upstreams live with the deployed
manifest.

## Development

```bash
cargo run -p ops-console            # needs the env above
cargo test -p ops-console           # session/CSRF/origin/XSS/no-secret tests
scripts/build-css.sh                # regenerate static/console.css (Tailwind v4 CLI)
```

`static/console.css` is generated from `style/input.css` and **checked in**
so builds never need node/tailwind. Regenerate it whenever a Tailwind class
is added/changed in `src/**` (the pre-commit `end-of-file-fixer` will touch
it; that's fine).

UI components are copy-paste vendored from the [Rust/UI registry]
(https://github.com/rust-ui/ui) (`app_crates/registry/src/ui/*.rs`), class
strings verbatim, `variants!`/`clx!` macros expanded to plain Leptos
components — see `src/ui.rs`.
