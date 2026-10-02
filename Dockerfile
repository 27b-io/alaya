# The compiler version comes from rust-toolchain.toml (COPY'd below; rustup
# installs it on first cargo call), so the base tag names none. Keep that file
# in the build context — .dockerignore must not exclude it. Both FROM lines pin
# a digest so one commit always builds on the same bases; Renovate refreshes it.
FROM docker.io/library/rust:bookworm@sha256:3ee46017ddbe6be5863d09382ba1ca613640dc115c0125b0d6ab56d2b967277c AS builder

# Build identity surfaced by GET /health/detail and MCP serverInfo (#70). Both are
# optional: an unset arg yields a null git_sha/built_at, never a build or
# startup failure. GIT_SHA must be the full 40-hex SHA to be reported.
ARG GIT_SHA=unknown
ARG BUILT_AT=

WORKDIR /app
COPY . .

# cmake is required to build aws-lc-sys (jsonwebtoken's aws_lc_rs crypto
# backend). Builder stage only — the final image copies just the binaries.
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

ENV ALAYA_GIT_SHA=${GIT_SHA}
ENV ALAYA_BUILT_AT=${BUILT_AT}
RUN cargo build --release -p alaya-bridge -p alaya-server -p ops-console

FROM docker.io/library/debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/alaya-bridge /usr/local/bin/
COPY --from=builder /app/target/release/alaya-server /usr/local/bin/
COPY --from=builder /app/target/release/ops-console /usr/local/bin/

EXPOSE 3000 3001 3002
