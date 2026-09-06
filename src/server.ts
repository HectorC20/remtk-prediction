/**
 * remtk-prediction — Entry del servidor HTTP (3 puertos).
 *
 * Arranca tres apps Nest autocontenidas (PredictAppModule 6776, EmbeddingAppModule
 * 6777, QdrantAppModule 6775) vía los helpers de `app.module.ts`. Separado de
 * `main.ts` (librería in-process) para que el paquete pueda usarse como librería
 * vía `src/index.ts` o como servicio con `node dist/src/server.js`.
 */
import { loadConfig } from "./config";
import { startEmbedServer, startPredictServer, startQdrantServer } from "./app.module";
import { log } from "./logger";

async function main(): Promise<void> {
  const cfg = loadConfig();
  log("boot", `config: predict=${cfg.portPredict} embed=${cfg.portEmbed} qdrant=${cfg.portQdrant}`);
  log("boot", `models: path=${cfg.onnxModelsPath} size=${cfg.onnxModelSize} onnx=${cfg.onnxEnabled}`);
  log("boot", `qdrant: enabled=${cfg.qdrantEnabled} url=${cfg.qdrantUrl}`);

  const [predict, embed, qdrant] = await Promise.all([
    startPredictServer(cfg.portPredict),
    startEmbedServer(cfg.portEmbed),
    startQdrantServer(cfg.portQdrant),
  ]);

  log("boot", `→ Predicción  http://localhost:${predict.port}`);
  log("boot", `→ Modelos     http://localhost:${embed.port}`);
  log("boot", `→ Qdrant      http://localhost:${qdrant.port}`);

  const shutdown = (): void => {
    log("boot", "apagando...");
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

if (require.main === module) {
  main().catch((err) => {
    console.error("[boot] error fatal:", err);
    process.exit(1);
  });
}
