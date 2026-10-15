# ops-console

OIDC-gated web console for the 27b workspace (LAB-1684 / LAB-1641). Three
route modules in one crate: Ālaya memory curation, the anthropic-lb
read-only monitoring pane (LAB-1964), and the selecta read-only pane
(LAB-6674).

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
| `SELECTA_URL` / `SELECTA_API_KEY` | selecta module — **both or neither**, same rules as the LB group. `SELECTA_URL` = selecta's HTTP base URL; its host:port is sent as the `Host` header, and selecta answers 421 to any Host outside its `allowed_hosts`, so use exactly the name selecta allow-lists. `SELECTA_API_KEY` = a selecta **read-only** token, sent server-side as `Authorization: Bearer` to `/readonly/mcp`. |
| `LB_URL` / `LB_API_KEY` / `METRICS_URL` | anthropic-lb module — **all three or none**. None: the module is disabled and the home card says so. A partial set refuses startup. `LB_URL` = the LB's base URL; `LB_API_KEY` = an LB **operator** client key, sent server-side as `x-api-key` (the LB rejects `Authorization: Bearer` on `/_stats`); `METRICS_URL` = a Prometheus-compatible query API for the 7-day history — the console calls `/api/v1/query_range` on it, plus the client-store module's per-metric routes when that module is on, so point it at a proxy that exposes nothing else. |
| `CLIENT_STORE_URL` / `CLIENT_STORE_TOKEN` | Client-store module — **both or neither**, and it needs the anthropic-lb module too. `CLIENT_STORE_URL` = the client store's origin, with no path (the console calls `{url}/_clients/admin`, nothing else); `CLIENT_STORE_TOKEN` = the store's **writer** token, at least 32 printable ASCII characters, sent server-side as `Authorization: Bearer`. |
| `CONSOLE_METRICS_ADDR` | Optional, default `0.0.0.0:9102`, an `ip:port` literal (no hostname, so loading the config needs no lookup), used only with the client-store module: the listener that serves `/metrics` (the committed-version labels) to the scraper. Keep it off any ingress that reaches the browser. |

Every upstream URL (`ALAYA_URL`, `LB_URL`, `METRICS_URL`, `SELECTA_URL`, `CLIENT_STORE_URL`) must be https, or plain http to a cluster-local host (`*.svc`, `*.svc.cluster.local`, `*.internal`, a single-label name, or a loopback/private IP literal), and must carry no query string or fragment. Anything else refuses startup — `METRICS_URL` included, keyless or not: off-cluster plaintext leaves the budget history both readable and rewritable in flight. `CONSOLE_OIDC_ISSUER` is stricter still: https only, no cluster-local exemption, and no userinfo (it is printed verbatim in the startup config log).

Logs are JSON, one object per line on stdout, so structured fields (`sub`,
`op`, `issuer`, `cause`, ...) are queryable as `fields.<name>` rather than
scraped from text. `RUST_LOG` defaults to `ops_console=info,tower_http=info`.
Untrusted values are recorded in Debug form, so their JSON string keeps the
quotes and escapes: a subject logs as `"sub":"\"user-123\""`, and the value
is `user-123`, not `"user-123"`.

### Allowlisting an admin

1. Have them log in once (they'll get a 403 page); the rejected subject is
   in `fields.sub` on the console's `login rejected: subject not
   allowlisted` record. It is a Debug string, so strip the inner quotes *and*
   decode its backslash escapes (`a\\b` is `a\b`) — `jq -r '.fields.sub |
   fromjson'` on the record does both. A quoted or still-escaped entry never
   matches and the 403 persists silently. (Or read the `sub` from the IdP's
   user admin.)
2. Add it to `CONSOLE_ALLOWED_SUBJECTS` (comma-separated) and roll the pod.

## Ālaya module

- **Browse/search** — the mode is picked by link, and its form offers only
  the filters the server applies in it: hybrid takes a query and a type,
  scan and recent a type, tag its tags; all four take the
  superseded-visibility toggle. A filter a request carries that its mode
  ignores is named in a "not applied" notice, never dropped silently, and
  hybrid says that its type filter reaches only the semantic matches. Paging
  comes only from the server's `has_more` / `next_cursor`. Hybrid ranks a
  bounded candidate pool, not the corpus, and its count line says "Top N
  candidates". Every search is sent with
  `read_only: true`, so browsing never bumps `access_count` or queues Hebbian
  co-access updates. That needs an alaya-server that knows the flag; an
  older one ignores it and writes. Both ship in the same image.
- **Detail** — full content, metadata, salience/access/trust stats,
  supersession chain (audit trail rendered, never hidden), and relations
  grouped by type and direction, each CONTRADICTS edge linking to its pair
  page. A failed relations read renders as "unavailable", never as
  "No relations.". A superseded memory shows what superseded it (a link)
  and the stored `supersession_reason`.
- **Curation** — supersede (reason required), correct-&-supersede (store a
  fixed copy, then supersede the original), delete (two-step confirm),
  merge duplicates (dry-run preview before commit), relations
  create/delete.
- **Un-supersede** — reverse a wrong supersession from the memory's detail
  page, reason required: posts `POST /unsupersede` with `unsuperseded_via:
  "operator:console"`. Each superseded hop in the chain links to its own
  form. The server's typed answers (*not superseded*, *superseded again*)
  show as messages, not errors.
- **Contradictions triage** — the CONTRADICTS queue, 50 pairs a page,
  newest first, paged with the server's `next_offset` (Next appears only
  when the server returns one). The view lives in the URL, so a refresh or
  a return keeps the operator's place, and other panes may link into it:
  `verdict` (repeatable: `contradiction`, `supersession`, `coexist`,
  `unrelated`, `unjudged`; absent = the server default), `resolved=1`
  (include resolved pairs) and `offset`. An unknown value is a 400. Each
  card shows the judge's verdict and confidence, the detector's confidence,
  the recommended survivor, the reason, model and timestamps, and a
  resolved pair's stamp. A stored judge failure renders as an error;
  a pair the judge never reached says so. *Review in full* opens
  `/alaya/contradictions/pair?a=…&b=…` (plus the queue view): both memories
  side by side in full, read live, with the verdict re-read from the queue
  page it was opened from. Decisions — *Keep A* / *Keep B* (supersede with
  a reason), *Keep both* (stamp the pair, nothing superseded), *Reopen*
  (clear a keep-both, in both edge directions) — all land back on the same queue view with a flash
  naming the pair. Tick cards and *Keep both for selected* settles a page in
  one submit, one resolution call per pair within a 30 s budget; the flash
  names every pair it could not confirm, and why. Supersede is never offered
  in bulk.
- **Auth state** — read-only view of alaya-server's `GET /auth/config`:
  principal × operation matrix + OIDC issuer/audience/allowlist.
- **Judge health** (`/alaya/health`) — read-only view of alaya-server's
  `GET /stats`: verdict mix (each count links to the contradictions
  filter), stored judge failures, judgements per day, degenerate reasons,
  daily-cap usage and graph edges by type. A source that is down renders
  as an "unavailable" banner, never as zeros.

## anthropic-lb module (read-only)

Scope: the console **renders** LB state; this module changes nothing.
Endpoints change through GitOps only, and so do the clients in the LB's own
config; the clients the client store holds change through the client-store
module below, which writes to the store, never to the LB. The console has
**no write route to the LB**, the LB exposes no admin write API, and this
module must not grow one. Values render with their provenance; limits are
"TOML, GitOps".

- **Fleet** — routing strategy, replicas seen, shared-state (Redis) health,
  cumulative upstream transport errors ranked by kind. Source:
  `GET /_stats`. Only fleet-wide values render: `/_stats` also carries
  process-local counters (per-consumer request rates, per-endpoint burn
  rates) that describe one random replica behind a Service, so they are
  deliberately left out.
- **Per-client budget burn** — one tile per client: today's fleet-wide used /
  limit as a meter (`/_stats` → `cluster.budget_usage`, the Redis aggregate)
  over a 7-day sparkline, the exact figures folded into a table below.
  When that aggregate is absent or empty the card falls back to the
  replica-local `client_budgets` mirror and says so with a "replica-local"
  badge — that mirror resets on pod restart and undercounts the fleet. Plus
  a 7-day history: the daily peak of `anthropic_cluster_budget_used` per
  UTC day, today's column running. History is one `query_range` against
  `METRICS_URL` (`max by (client) (max_over_time(…[23h58m]))` at 23:59 UTC
  of each day — the trimmed window keeps the first post-midnight scrape,
  which can still carry yesterday's total, out of today's peak). No history
  store in the console, no dashboard embeds.
- **Upstream accounts** — per endpoint: 5-hour / 7-day window utilisation as
  meters (amber from 0.80, red from 0.98: the floors the LB reads
  `allowed_warning` and `throttled` as), hard-limit / throttle status, next
  5h reset; hottest first.

Each card degrades on its own: a dark metrics store leaves the live budget
meters up, an LB outage leaves the burn history up. Every upstream client in the
console refuses redirects (no credential ever rides a 3xx off-host).

## Client-store module (LB client management)

The LB's ordinary clients live in a client store that every LB instance
polls: a document that grants (clients, keys, policies) and a set of
restrictions that deny (disabled names, revoked key digests). The console
manages them through the store's admin API, never through the LB: `lb.rs`
only ever GETs, and `no_route_sends_anything_but_a_get_to_the_lb` drives
every route to prove it.

Two kinds of write, both at `/lb/clients`.

**The kill switch: restriction ops.**

- **`disable` / `enable`** by name, each with a reason, sent as restriction
  ops. `disable` refuses every key of the name, the TOML keys of an ordinary
  client included; `enable` carries the version of the record it read, so a
  newer `disable` gets 409 instead of being undone. Nothing read from the
  metrics store is consulted before either is sent. `/lb/clients/disable`
  loads the form without reading anything, for when the store or the metrics
  store is slow.
- **A full disabled set.** The store refuses a request that leaves more than
  1,024 names disabled and more than it found; the form then offers disabled
  names to enable in the same request, those the readers report matching no
  client first.

**Document ops: clients, policies and keys.** `create`, `update` (all four
policy fields: daily budget, utilization limit, models, pinned endpoints),
`delete` (its keys stay unrevoked, so `restore` can undo it), `add_key`,
`revoke_key` (by digest, for good) and `restore` (from history, at
`/lb/clients/history`).

- **Preview, then commit.** Every form posts to `/lb/clients/preview`, which
  sends the op as a `dry_run` pinned to the document version the page read
  and shows the store's diff. The commit form at `/lb/clients/commit` carries
  the same op and the same version, so a resubmitted form gets 409 from the
  store and never writes twice. After any 409 the console re-reads the store
  and says what it holds.
- **The key page.** `add_key`'s commit answers with the key page itself, not a
  redirect: the key the store minted, shown once, sent `Cache-Control:
  no-store`. It never goes in the flash cookie, a URL or a log line: the write
  log keeps only the answer members it knows, and each key's client, id and
  digest. Reloading it re-sends the commit, which the store refuses with 409.
  A preview whose answer is not a dry run's is reported as an unknown outcome
  and logged, never rendered. Rotation with overlap: add a key, move consumers
  to it, then revoke the old one.
- **Typed confirmations.** While any instance reports the store behind or
  edited (`store_behind`, `store_edited`), every document write needs `write
  anyway` typed: the store was rolled back or edited outside its write path,
  and a write would publish that content at a newer version. The same holds
  while that cannot be told: the metrics store does not answer, shows no
  reader up, or shows a reader that is up but does not export both of those
  alerts, which a reader always exports, raised or not. A write still goes
  ahead once the phrase is typed.
- **The `restore` row check.** The store's dry run names the digest of the
  history row it would restore, and the commit sends that digest back, so the
  store refuses the restore, inside the commit's transaction, if the row
  changed after the preview. The digest is checked against the (version,
  digest) pairs instances installed in the last 30 days. A version installed
  with another digest means the row was edited after the fact, and the
  restore is refused. A version with no install on record needs its version
  typed. An empty answer from the metrics store is unknown, and the restore is
  refused. The check runs again at commit. A store whose dry run names no
  digest cannot bind a restore, so the console offers none.
- **TOML warnings.** A `delete` or `restore` preview names each client it
  drops that a stack's TOML still holds: on that stack the name goes back to
  its TOML policy and keys.

**In effect, per stack.** After a write, `/lb/clients/effect` shows it on
each stack that counts, refreshing until every stack holds it (or two
minutes pass). CF is judged from the store's report of CF's latest poll; every
other stack from its instances in the metrics store, each of which must be
listed by `up`, be up and export exactly one series of the kind judged. A
restriction op is in effect on an instance when the instance reports the set
hash of the restrictions part the store serves now, and that part carries the
op's record or a newer one of the same state. A document is in effect when the
instance holds its digest and the store still serves it. What moved between
the two store reads around the metrics read confirms nothing, and neither does
a partial restrictions part. A newer record in the other state, or a later
document, reads as superseded, never as in effect. The rules live in
`effect.rs`; the stacks card on `/lb/clients` shows both for what the store
serves now.

**Warnings.** A partial restrictions part, instances raising
restrictions-behind or reporting the store behind or edited, a frozen or
unreadable document, a refused recovery, an open point-in-time-recovery
window, and disabled names the readers report matching nothing ("mistyped?").

Every write route in the console, the Ālaya memory-curation ones included,
re-checks the session's subject against `CONSOLE_ALLOWED_SUBJECTS` (otherwise
read only at login), after the CSRF token and the Origin check. Every
client-store attempt that reaches its handler is logged on target
`ops_console::write_log` with the subject: a refusal made before sending
(CSRF, allowlist, malformed input, a missing confirmation), the request with
the store's answer, or the upstream failure. A request without the console's
own Origin is refused earlier, by middleware, before any session is read. A
write is sent, logged and exported as committed in a task of its own, so a
dropped connection cannot lose its log line or its labels. An answer that is
lost, or any status but 200 or a 4xx refusal, leaves the outcome unknown: the
console says so, re-reads the store, and exports nothing as committed.

**What the metrics proxy must serve.** One route per metric, at
`{METRICS_URL}/m/<metric>/api/v1/query`, each pinning `__name__` (and the
LB's scrape job) with `extra_label` so a query can only narrow it:
`up`, `anthropic_client_store_restrictions`, `anthropic_client_store_document`,
`anthropic_client_store_alert`, `anthropic_client_store_unmatched_disabled_name`
and `anthropic_client_store_toml_client`. Instant queries only: the 30-day
window the `restore` check needs is a range selector inside the query. A query
outside a pin comes back empty, which the console reads as "nothing visible",
never as healthy. Each reader stack's series must carry `stack="<name>"`, the
name the store gives it (its reader role, `reader:<name>`); the CF scrape's
`stack="cloudflare"` is read as the store's `cf`. A reader series the console
cannot place, with no `stack` or no pod or instance label, or of a stack the
store does not count, leaves every stack judged from the metrics store
unknown: dropped, it would let its stack read as in effect without it.

**Committed labels.** On `CONSOLE_METRICS_ADDR` the console serves
`ops_console_committed_record{name,version}`, one series per restriction
record it committed, and `ops_console_committed_document{version,digest}`, one
per document (the value is when). An alert can match the versions instances
install against these over 30 days and fire on one the console never wrote.
The lists are in memory: a commit the scraper never saw, because the console
restarted within one scrape interval, reads as unmatched, and the write log
holds it.

## selecta module (read-only)

Scope: the console **renders** what selecta is doing; nothing in it changes
selecta. The console has **no write route to selecta** and this module must
not grow one. Approvals render **only as links to their GitHub pull
request**, never as a button: merging that PR is selecta's only approval
channel. The guarantee also holds at selecta, not just here: the console's
token is a selecta read-only token, accepted only on the `/readonly/mcp`
mount, which registers `selecta_list_tasks`, `selecta_get_task`,
`selecta_download_health` and `selecta_list_queues` and no verb that
enqueues work. Enqueue actions from the console would need their own
security review first.

`/selecta` shows, each card from its own source:

- **Runner** — heartbeat (selecta's own `GET /healthz` verdict: advancing,
  or stalled with its reason), lease (held with seconds left, or not held:
  selecta then refuses irreversible work), run budget (parked or not, with
  selecta's reason) and what the run consumed, and the daily caps used
  against their limits. Source: `GET /metrics`. selecta does not export its
  run caps; a cap shows inside the parked reason once it is hit.
- **Waiting on you** — tasks in `awaiting_approval`, newest first, each with
  its approval PR link. A task with no PR cannot be approved at all and is
  flagged as an error. Only a URL of the form
  `https://github.com/<owner>/<repo>/pull/<n>` becomes a link.
- **Recent failures** — the newest failed tasks: verb, enqueued time and the
  200-character audit excerpt, each linking to `/selecta/task/<id>`, which
  renders `selecta_get_task` with the whole audit.
- **Tasks by state** — `queue_depth` per state: a stacked bar of the tasks
  still in flight (`queued` through `executing`) over a tile per state. The
  tiles count all time: selecta never deletes a queue row, so `done`,
  `aborted_stale` and `failed` only grow, and stay off the bar. A state the
  console does not know still gets a tile, and is flagged as an error.
- **Download health** — `selecta_download_health` signals and alerts.

Each read is one stateless JSON-RPC `tools/call` POST (no `initialize`),
parsed from `structuredContent`. **Fail loud**: a tool error, or a payload
or metric the pane cannot parse, renders as an error naming the tool or
source — never as an empty list, which would read as "all quiet".

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
`LB_URL`, `METRICS_URL`, `SELECTA_URL` are plain env in the manifest;
`CONSOLE_OIDC_CLIENT_ID`, `CONSOLE_OIDC_CLIENT_SECRET`,
`CONSOLE_ALLOWED_SUBJECTS`, `CONSOLE_SESSION_SECRET`, `ALAYA_API_KEY`,
`LB_API_KEY` and `SELECTA_API_KEY` come from a secret manager, rendered by
ESO into Secret `ops-console-env`. Editing the Secret rolls the pod (Reloader annotation).
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

The equivalent probes for the LB and selecta panes' upstreams live with the
deployed manifest.

## Development

```bash
cargo run -p ops-console            # needs the env above
cargo test -p ops-console           # session/CSRF/origin/XSS/no-secret tests
scripts/build-css.sh                # regenerate static/console.css (Tailwind v4 CLI)
scripts/snapshot.sh                 # screenshot every page: 390/768/1280px, light and dark
```

`static/console.css` is generated from `style/input.css` and **checked in**
so builds never need node/tailwind. Regenerate it whenever a Tailwind class
is added/changed in `src/**` (the pre-commit `end-of-file-fixer` will touch
it; that's fine). Nothing else catches a stale file: Tailwind emits only the
classes it finds in `src/`, so a class added without a rebuild ships with no
rule and its element renders unstyled, silently.

**Presentation framework (`src/ui/`).** Pages compose components; they do
not hand-write layout or recurring markup.

- `ui/components.rs`: the atoms (button, badge, card, table, form classes),
  copy-paste vendored from the Rust/UI registry, class strings verbatim
  except the departures marked in `components.rs`,
  `variants!`/`clx!` macros expanded to plain Leptos components. The Leptos
  registry now lives at [rust-ui/leptos-ui](https://github.com/rust-ui/leptos-ui)
  (`app_crates/registry/src/ui/*.rs`, theme tokens in `style/tailwind.css`);
  [rust-ui/ui](https://github.com/rust-ui/ui) has moved to Dioxus.
- `ui/patterns.rs`: the console's own components. `Status` maps what a
  state means to its badge;
  `Notice` is a status badge plus a sentence (an `Error` notice is an alert),
  and `unavailable()` is the notice for a dark section; `EmptyState`,
  `NotConfigured`, `Stat`/`StatGrid`, `PageHeader` (each page has exactly
  one), `Link` (the one link style; `external` sends no Referer) and
  `HashLink`.
- `ui/charts.rs`: key values as server-rendered SVG, which the CSP allows
  where a chart library's script would not: `Meter` (used against a limit,
  with warning and critical thresholds), `StackedBar` (a part-to-whole
  mix), `BarSeries` (bars per day, optionally stacked), `Sparkline` (a trend
  under a `Stat`'s value) and `HBarList` (a ranking). A chart is one
  `role="img"` element with an `aria-label` that states its numbers, each
  mark has a `<title>` tooltip, and the numbers are also text: beside the
  marks, or, for the daily bars and the sparkline, in a visually hidden
  table. A missing, NaN or negative value is "no data", never a zero.
  Colours are `Colour` constants named by their job; `style/input.css`
  holds their values and how they were checked.
- `ui/shell.rs`: the document and the page shell. Every handler passes the
  `Nav` entry its page belongs under to `page()`; the shell marks that entry
  `aria-current="page"`. From `lg` the navigation is a sidebar grouped by
  module; below it, a `<details>` menu in the top bar. The top bar holds the
  theme switch and logout once at every width.

A wide table keeps the columns that say what a row is and how it stands; the
rest are `secondary` on `TableHead` and `TableCell` and show from `md` up. A
form stacks its fields below `sm` and lines them up from there.

The CSP forbids scripts and inline style, so nothing may emit a script, a
style element or a style attribute, and colour comes only from the theme
tokens, never a Tailwind palette class (`text-white`, `bg-gray-100`). Tests
in `src/ui/mod.rs` hold the production code to the first, and every file
under `src/` (tests included: Tailwind reads them all) and the checked-in
stylesheet to the second. Tailwind reads `src/` and nothing else
(`source(none)` in `style/input.css`), so prose like this paragraph never
reaches the stylesheet.

**Snapshots.** `scripts/snapshot.sh` renders every GET page against the test
suite's fake upstreams (an ignored test, `tests::snapshot::write_snapshots`),
both healthy and with every upstream down, then screenshots each with
headless Chrome at 390, 768 and 1280px, light and dark, into
`target/snapshots/` at the workspace root, with one contact sheet per page.
It fails if any page scrolls sideways; a table may, inside its own scroll
box. The first sheet, `charts`, is a gallery of every chart in its normal
and edge states; no route serves that page. A UI change attaches the
sheets to its review. Needs python3 and Chrome (`CHROME=/path/to/chrome`
if it is not on `PATH`).

**Theme.** Every colour is a CSS variable in `style/input.css`, defined once
for light and once under `@variant dark`; components use the token classes
(`bg-card`, `text-muted-foreground`, …), never a palette colour, so a new
component is dark-mode-ready by construction. A new token needs a value in
both blocks and a `--color-*` line in `@theme inline`. The theme switch in the
top bar saves the operator's choice in the encrypted `console_theme`
cookie (`src/theme.rs`); the page renders it as `class="dark"` or
`class="light"` on `<html>`, and with no choice saved the OS setting
(`prefers-color-scheme`) decides. No JavaScript is involved. The switch
returns to the page's navigation entry, which the form carries as `back`: a
short GET page, so a switch used on a POST's answer page lands on a page too.
