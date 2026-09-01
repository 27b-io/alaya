FROM docker.io/library/rust:1.87-bookworm AS builder

# Build identity surfaced by GET /health and MCP serverInfo (#70). Both are
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

FROM docker.io/library/debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/alaya-bridge /usr/local/bin/
COPY --from=builder /app/target/release/alaya-server /usr/local/bin/
COPY --from=builder /app/target/release/ops-console /usr/local/bin/

EXPOSE 3000 3001 3002
