# Instalación 02: Configuración en TypeScript (`pnpm`), Docker y Producción

Esta guía detalla cómo configurar el proyecto principal `remtk-prediction` para ejecutar el modelo `remtk-juicio-moe-int8` junto con `e5-small`/`e5-large` y Qdrant en los 3 puertos (`6776`, `6777`, `6775`).

---

## 1. Instalación de Dependencias en Node.js (`pnpm`)

En la raíz de `remtk-prediction`, asegúrate de contar con `onnxruntime-node` para la ejecución nativa en C++ del grafo MoE:

```bash
# Instalar dependencias del proyecto y asegurar onnxruntime-node
pnpm install
pnpm add onnxruntime-node@^1.19.2
```

---

## 2. Estructura Final de la Carpeta `models/`

Verifica que tu directorio `./models` (o la ruta configurada en `ONNX_MODELS_PATH`) tenga la siguiente disposición de archivos antes de iniciar el servidor:

```text
models/
├── multilingual-e5-small-onnx/
│   ├── model.onnx               # o model_quantized.onnx (384 dims)
│   ├── tokenizer.json
│   └── config.json
├── multilingual-e5-large-onnx/
│   ├── model.onnx               # (1024 dims para llamadas explícitas en Puerto 6777)
│   ├── tokenizer.json
│   └── config.json
└── remtk-juicio-moe-int8/
    ├── model_quantized.onnx     # (~18.5 MB - Único decisor para src/juicio/)
    └── config.json              # Metadatos de versión, umbrales y nombres de los 6 expertos
```

---

## 3. Configuración Recomendada de `.env` para Producción

```ini
# Puertos HTTP del Servicio Unificado
PORT_PREDICT=6776
PORT_EMBED=6777
PORT_QDRANT=6775

# Motor ONNX
ONNX_ENABLED=true
ONNX_MODELS_PATH=./models
MOE_MODEL_SUBPATH=remtk-juicio-moe-int8/model_quantized.onnx

# Calibración del Decisor y Umbral Adaptativo (Etapa B)
ONNX_ADAPTIVE_MIN_TOOLS=2
ONNX_ADAPTIVE_MAX_TOOLS=30
ONNX_ADAPTIVE_GAP_THRESHOLD=0.03
ONNX_ADAPTIVE_MIN_SCORE=0.80
MAX_CATEGORIES=60
MAX_OUTPUT_TOOLS=150

# Recall Léxico y Vectorial (Puerto 6775 -> Qdrant 6333)
QDRANT_ENABLED=true
QDRANT_URL=http://localhost:6333
RECALL_LIMIT=50
KEYWORD_BOOST=0.15
KEYWORD_TOP_K=20
PREDICT_WARMUP_WAIT_MS=15000
```

---

## 4. Optimización del `Dockerfile` para VPS Linux en Producción

Actualiza tu `Dockerfile` para copiar únicamente los modelos activos (`e5-small`, `e5-large` y `remtk-juicio-moe-int8`), excluyendo los modelos deprecados (`mDeBERTa` y `ms-marco-MiniLM`) y la carpeta `training/`:

```dockerfile
FROM node:20-bookworm-slim AS builder
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY package.json pnpm-lock.yaml tsconfig.json ./
RUN pnpm install --frozen-lockfile
COPY src/ ./src/
RUN pnpm build

FROM node:20-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
# Limitar consumo de hilos de ONNX Runtime para no saturar el Event Loop de Node
ENV OMP_NUM_THREADS=4

RUN corepack enable && corepack prepare pnpm@latest --activate
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --prod --frozen-lockfile

COPY --from=builder /app/dist ./dist
COPY models/multilingual-e5-small-onnx ./models/multilingual-e5-small-onnx
COPY models/multilingual-e5-large-onnx ./models/multilingual-e5-large-onnx
COPY models/remtk-juicio-moe-int8 ./models/remtk-juicio-moe-int8

EXPOSE 6775 6776 6777
CMD ["node", "dist/src/server.js"]
```

---

## 5. Comandos de Verificación Rápida Post-Instalación

```bash
# 1. Compilar y levantar el servicio
pnpm build && pnpm start

# 2. Verificar salud de los 3 puertos y carga del modelo MoE
curl http://localhost:6776/debug | jq '.engine, .stats'

# 3. Correr batería completa de tests E2E y de modelos reales
pnpm test:all
```