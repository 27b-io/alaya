# ops-console

OIDC-gated web console for the 27b workspace (LAB-1684 / LAB-1641). The
Ālaya memory-curation module ships first; the anthropic-lb read-only
monitoring pane lands as a second route module (LAB-1964) in this same crate.

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
- Ālaya's own OIDC principals stay **read-only**; this console adds zero new
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

### Allowlisting an admin

1. Have them log in once (they'll get a 403 page); the rejected `sub` is in
   the console log line `login rejected: subject not allowlisted`.
   (Or read the `sub` from the IdP's user admin.)
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

## Deploy

Deployed from the lab repo: `27b-io/lab` `k8s/mcp/ops-console.yaml` +
`ops-console-externalsecret.yaml` (LAB-2712). `deploy/console/ops-console.yaml`
here is a mirror of that manifest — keep them in sync. Shape: Deployment +
Service + NetworkPolicy — own label, egress pinned to alaya-server + IdP :443
(DNS via the namespace `allow-dns` policy), **no dragonfly egress**,
`imagePullSecrets: ghcr-creds` (the package is private), image digest-pinned
via the `flux-system:alaya` imagepolicy marker so the console rolls with
alaya-server. The binary ships in the existing `ghcr.io/27b-io/alaya` image
(`command: ["ops-console"]`).

Config split: `CONSOLE_PUBLIC_URL`, `CONSOLE_OIDC_ISSUER`, `ALAYA_URL` are plain
env in the manifest; `CONSOLE_OIDC_CLIENT_ID`, `CONSOLE_OIDC_CLIENT_SECRET`,
`CONSOLE_ALLOWED_SUBJECTS`, `CONSOLE_SESSION_SECRET` come from the 1Password
item `ops-console` (fields `oidc-client-id`, `oidc-client-secret`,
`allowed-subjects`, `session-secret`) and `ALAYA_API_KEY` from
`alaya-config/credential`, rendered by ESO into Secret `ops-console-env`.
Editing the Secret rolls the pod (Reloader annotation).

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
