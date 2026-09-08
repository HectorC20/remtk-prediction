# node:22-slim (glibc) — onnxruntime-node y @huggingface/tokenizers publican
# binarios glibc y NO cargan en Alpine (musl): "ld-linux-x86-64.so.2" ausente.
FROM node:22-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@9.15.9

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY tsconfig.json ./
COPY src ./src
RUN pnpm build

FROM node:22-slim

WORKDIR /app

RUN npm install -g pnpm@9.15.9

# libgomp1: OpenMP de ONNX Runtime (multi-hilo) sobre glibc.
# ca-certificates: evita el warning de telemetría de ORT ("No readable CA
# bundle was found") — sin él, ORT no puede validar HTTPS para telemetría.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY scripts/ensure-onnx-models.mjs ./scripts/ensure-onnx-models.mjs

# Modelos ONNX: bind mount ./models:/app/models (nunca en la imagen)
ENV ONNX_MODELS_PATH=/app/models
ENV NODE_ENV=production

EXPOSE 6776 6777 6775

# Descarga los modelos ONNX (idempotente, no bloqueante) y arranca el servidor.
CMD ["sh", "-c", "node scripts/ensure-onnx-models.mjs && node dist/src/server.js"]
