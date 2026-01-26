# Multi-stage Dockerfile for p2a-mcp backend
# Build: docker build -f docker/backend.Dockerfile -t p2a-mcp .

# Stage 1: Build
# Requires Rust 1.88+ for latest dependency compatibility
FROM rust:latest AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest files first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Copy all workspace member Cargo.toml files (needed for workspace resolution)
COPY crates/p2a-core/Cargo.toml ./crates/p2a-core/
COPY crates/p2a-mcp/Cargo.toml ./crates/p2a-mcp/
COPY crates/p2a-cli/Cargo.toml ./crates/p2a-cli/
COPY crates/p2a-dioxus/Cargo.toml ./crates/p2a-dioxus/

# Create dummy source files for all workspace members
RUN mkdir -p crates/p2a-core/src crates/p2a-mcp/src crates/p2a-cli/src crates/p2a-dioxus/src && \
    echo "pub fn dummy() {}" > crates/p2a-core/src/lib.rs && \
    echo "fn main() {}" > crates/p2a-mcp/src/main.rs && \
    echo "fn main() {}" > crates/p2a-cli/src/main.rs && \
    echo "fn main() {}" > crates/p2a-dioxus/src/main.rs

# Build dependencies only (cached layer) - this may fail but will cache deps
RUN cargo build --release -p p2a-mcp --features full 2>/dev/null || true

# Copy actual source code for the crates we need
COPY crates/p2a-core ./crates/p2a-core
COPY crates/p2a-mcp ./crates/p2a-mcp

# Touch the source files to invalidate the cache for the actual build
RUN touch crates/p2a-core/src/lib.rs crates/p2a-mcp/src/main.rs

# Build the actual binary
RUN cargo build --release -p p2a-mcp --features full

# Stage 2: Runtime
# Use Debian Trixie (testing) to match the glibc version from rust:latest
FROM debian:trixie-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    fontconfig \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash p2a

# Copy the binary
COPY --from=builder /app/target/release/p2a-mcp /usr/local/bin/

# Set ownership
RUN chown p2a:p2a /usr/local/bin/p2a-mcp

# Switch to non-root user
USER p2a
WORKDIR /home/p2a

# Expose the HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["p2a-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]
