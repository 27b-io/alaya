FROM docker.io/library/rust:1.87-bookworm AS builder

ARG GIT_SHA=unknown

WORKDIR /app
COPY . .

ENV ALAYA_GIT_SHA=${GIT_SHA}
RUN cargo build --release -p alaya-bridge -p alaya-server

FROM docker.io/library/debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/alaya-bridge /usr/local/bin/
COPY --from=builder /app/target/release/alaya-server /usr/local/bin/

EXPOSE 3000 3001
