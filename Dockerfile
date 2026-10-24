# ==============================================================================
# remtk-prediction — Servidor de Inferencia y Juicio de Herramientas en Rust Nativo
# Reemplazo de alta eficiencia (< 88 MB RSS) para la antigua arquitectura NestJS
# ==============================================================================

# ==============================================================================
# STAGE 1: Builder (Compilación de Rust y Linkeo con ONNX Runtime)
# ==============================================================================
FROM rust:bookworm AS builder

WORKDIR /app

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Descargar la biblioteca compartida oficial de Microsoft ONNX Runtime v1.20.1
ENV ORT_VERSION=1.20.1
RUN curl -sSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    | tar -xz --strip-components=1 -C /usr/local/

COPY Cargo.toml ./
COPY .cargo .cargo
COPY src src

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Compilar en modo release con LTO y optimizaciones AVX2
RUN cargo build --release

# Strip del binario principal para remover símbolos de depuración
RUN strip /app/target/release/remtk-prediction-server

# ==============================================================================
# STAGE 2: Runtime Ultraligero (Debian Bookworm Slim - < 45 MB)
# ==============================================================================
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar bibliotecas de ONNX Runtime
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/

# Copiar el binario compilado nativo en Rust
COPY --from=builder /app/target/release/remtk-prediction-server /app/remtk-server

# Variables de entorno canónicas
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PORT_PREDICT=6776
ENV PORT_EMBED=6777
ENV PORT_QDRANT=6775
ENV MODELS_PATH=/app/models
ENV RUST_LOG=info

EXPOSE 6776 6777 6775

ENTRYPOINT ["/app/remtk-server"]
