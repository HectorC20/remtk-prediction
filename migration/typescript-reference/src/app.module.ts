/**
 * remtk-prediction — Adaptador HTTP ligero nativo (sin NestJS).
 *
 * Mantiene compatibilidad con las suites de tests en TypeScript (startPredictServer,
 * startEmbedServer, startQdrantServer) sin requerir el runtime pesado de NestJS/Express.
 * El servidor oficial de producción es ahora el binario nativo en Rust (rust-server).
 */
import http from "node:http";
import { URL } from "node:url";
import { loadConfig } from "./config";
import { createSystem, type System } from "./main";
import { SpatialPredictService } from "./predict/services/spatial-predict.service";
import { decidePlannerFastPath } from "./predict/services/planner-fast-path.service";

export interface ServerHandle {
  port: number;
  close(): Promise<void>;
}

export interface EmbedServerHandle extends ServerHandle {}

let sharedSystem: System | null = null;
let sharedSpatial: SpatialPredictService | null = null;

async function getSystem(): Promise<{ system: System; spatial: SpatialPredictService }> {
  if (!sharedSystem) {
    const cfg = loadConfig();
    sharedSystem = await createSystem(cfg);
    sharedSpatial = new SpatialPredictService(sharedSystem.engine);
  }
  return { system: sharedSystem, spatial: sharedSpatial! };
}

function startHttp(
  port: number,
  handler: (
    req: http.IncomingMessage,
    res: http.ServerResponse,
    body: any,
    system: System,
    spatial: SpatialPredictService,
  ) => Promise<boolean>,
): Promise<ServerHandle> {
  return new Promise(async (resolve, reject) => {
    try {
      const { system, spatial } = await getSystem();
      const server = http.createServer((req, res) => {
        let raw = "";
        req.on("data", (chunk) => {
          raw += chunk;
        });
        req.on("end", async () => {
          let body = {};
          try {
            body = raw ? JSON.parse(raw) : {};
          } catch {}
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Access-Control-Allow-Methods", "*");
          res.setHeader("Access-Control-Allow-Headers", "*");

          if (req.method === "OPTIONS") {
            res.writeHead(204);
            return res.end();
          }

          if (req.url === "/health") {
            res.writeHead(200);
            return res.end(JSON.stringify({ status: "ok" }));
          }

          try {
            const handled = await handler(req, res, body, system, spatial);
            if (!handled) {
              res.writeHead(404);
              res.end(JSON.stringify({ error: "not found" }));
            }
          } catch (err: any) {
            res.writeHead(500);
            res.end(JSON.stringify({ error: String(err?.message || err) }));
          }
        });
      });

      server.listen(port, () => {
        const addr = server.address() as any;
        resolve({
          port: addr?.port ?? port,
          close: () => new Promise<void>((r) => server.close(() => r())),
        });
      });
      server.on("error", reject);
    } catch (err) {
      reject(err);
    }
  });
}

/** Arranca el servidor de predicción (puerto 6776 o efímero). */
export function startPredictServer(port: number): Promise<ServerHandle> {
  return startHttp(port, async (req, res, body, system, spatial) => {
    const url = req.url || "/";
    if (url.startsWith("/tools/count")) {
      const u = new URL(url, "http://localhost");
      const tenant = u.searchParams.get("tenant") || "default";
      const count = system.orchestrator.countTools(tenant);
      res.writeHead(200);
      res.end(JSON.stringify({ count }));
      return true;
    }
    if (url === "/tools" && req.method === "POST") {
      const indexed = await system.orchestrator.registerTools(body.tenant, body.tools);
      res.writeHead(200);
      res.end(JSON.stringify(indexed));
      return true;
    }
    if (url === "/predict" && req.method === "POST") {
      const result = await system.orchestrator.predict(body);
      res.writeHead(200);
      res.end(JSON.stringify(result));
      return true;
    }
    if (url === "/predict/route" && req.method === "POST") {
      const decision = decidePlannerFastPath(body);
      res.writeHead(200);
      res.end(JSON.stringify(decision));
      return true;
    }
    if ((url === "/predict/spatial" || url === "/spatial/predict") && req.method === "POST") {
      const result = await spatial.predict(body);
      res.writeHead(200);
      res.end(JSON.stringify(result));
      return true;
    }
    if (url === "/memory/predict" && req.method === "POST") {
      const result = await system.orchestrator.predictMemory(body);
      res.writeHead(200);
      res.end(JSON.stringify(result));
      return true;
    }
    if (url === "/memory/judge" && req.method === "POST") {
      const text = String(body?.text ?? "").trim();
      const result = await system.orchestrator.judgeMemory(text);
      res.writeHead(200);
      res.end(JSON.stringify(result));
      return true;
    }
    if (url.startsWith("/debug") && req.method === "GET") {
      const engineInfo = await system.engine.getInfo(system.engine.defaultSize);
      const snapshot = system.debugger.snapshot();
      res.writeHead(200);
      res.end(
        JSON.stringify({
          engine: { size: engineInfo.model, modelPath: engineInfo.modelPath },
          stats: snapshot.stats,
          traces: snapshot.traces,
        }),
      );
      return true;
    }
    return false;
  });
}

/** Arranca el servidor de modelos y embeddings (puerto 6777 o efímero). */
export function startEmbedServer(port: number): Promise<ServerHandle> {
  return startHttp(port, async (req, res, body, system) => {
    const url = req.url || "/";
    if (url === "/embed" && req.method === "POST") {
      const size = body.size || system.engine.defaultSize;
      const r = await system.engine.embed(body.text, size);
      res.writeHead(200);
      res.end(
        JSON.stringify({
          embedding: Array.from(r.embedding),
          model: r.model,
          dim: r.dim,
          modelPath: r.modelPath,
        }),
      );
      return true;
    }
    if (url === "/models" && req.method === "GET") {
      const size = system.engine.defaultSize;
      const info = await system.engine.getInfo(size);
      res.writeHead(200);
      res.end(JSON.stringify({ [size]: info }));
      return true;
    }
    return false;
  });
}

/** Arranca el servidor de Qdrant (puerto 6775 o efímero). */
export function startQdrantServer(port: number): Promise<ServerHandle> {
  return startHttp(port, async (req, res, body, system) => {
    const url = req.url || "/";
    if (url === "/status") {
      const s = await system.qdrant.status();
      res.writeHead(200);
      res.end(JSON.stringify(s));
      return true;
    }
    if (url === "/tools/upsert" && req.method === "POST") {
      await system.qdrant.indexTools(body.tenant, body.tools);
      res.writeHead(200);
      res.end(JSON.stringify({ indexed: system.qdrant.countTools(body.tenant) }));
      return true;
    }
    if (url === "/tools/search" && req.method === "POST") {
      const tools = await system.qdrant.retrieveTools(body.tenant, body.text, body.limit);
      res.writeHead(200);
      res.end(JSON.stringify({ tools }));
      return true;
    }
    if (url.startsWith("/tools/count")) {
      const u = new URL(url, "http://localhost");
      const tenant = u.searchParams.get("tenant") || "default";
      res.writeHead(200);
      res.end(JSON.stringify({ count: system.qdrant.countTools(tenant) }));
      return true;
    }
    return false;
  });
}
