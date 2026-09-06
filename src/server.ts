/**
 * remtk-prediction — Entry del servidor HTTP (3 puertos).
 *
 * Separado de `main.ts` (librería in-process) para que el paquete pueda usarse
 * como librería vía `src/index.ts` o como servicio con `node dist/src/server.js`.
 */
import { loadConfig } from "./config";
import { createEmbedServer } from "./embedding/embed-server";
import { log } from "./logger";
import { createPredictServer } from "./predict/predict-server";
import { createQdrantServer } from "./qdrant/qdrant-server";
import { createSystem } from "./main";

function listen(app: import("express").Express, port: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = app.listen(port, () => {
      const addr = server.address();
      const actual = typeof addr === "object" && addr ? addr.port : port;
      resolve(actual);
    });
    server.on("error", reject);
  });
}

async function main(): Promise<void> {
  const cfg = loadConfig();
  log("boot", `config: predict=${cfg.portPredict} embed=${cfg.portEmbed} qdrant=${cfg.portQdrant}`);
  log("boot", `models: path=${cfg.onnxModelsPath} size=${cfg.onnxModelSize} onnx=${cfg.onnxEnabled}`);
  log("boot", `qdrant: enabled=${cfg.qdrantEnabled} url=${cfg.qdrantUrl}`);

  const system = await createSystem(cfg);

  const predictApp = createPredictServer(system.orchestrator, system.debugger, system.engine);
  const embedApp = createEmbedServer(system.engine);
  const qdrantApp = createQdrantServer(system.qdrant);

  const [predictPort, embedPort, qdrantPort] = await Promise.all([
    listen(predictApp, cfg.portPredict),
    listen(embedApp, cfg.portEmbed),
    listen(qdrantApp, cfg.portQdrant),
  ]);

  log("boot", `→ Predicción  http://localhost:${predictPort}`);
  log("boot", `→ Modelos     http://localhost:${embedPort}`);
  log("boot", `→ Qdrant      http://localhost:${qdrantPort}`);

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
