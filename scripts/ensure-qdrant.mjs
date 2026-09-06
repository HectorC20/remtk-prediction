/**
 * ensure-qdrant.mjs: garantiza que el Qdrant local (docker-compose.yml de este
 * proyecto) esté disponible antes de `pnpm dev`.
 *
 * - Si el puerto de QDRANT_URL ya responde → no hace nada (exit 0).
 * - Si está cerrado → levanta el contenedor `qdrant` con `docker compose up -d`.
 * - Si el daemon de Docker está apagado → intenta abrir Docker Desktop y espera.
 * - Guarda: con QDRANT_ENABLED=false se omite por completo.
 */
import { spawn, spawnSync } from "node:child_process";
import net from "node:net";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const composeFile =
  process.env.QDRANT_COMPOSE_FILE || path.join(rootDir, "docker-compose.yml");

const DEFAULT_QDRANT_URL = "http://localhost:6333";
const CONTAINER_WAIT_MS = 30_000;
const DOCKER_DESKTOP_WAIT_MS = 90_000;

function qdrantTarget() {
  const raw = process.env.QDRANT_URL || DEFAULT_QDRANT_URL;
  const url = raw.includes("://") ? new URL(raw) : new URL(`http://${raw}`);
  return { host: url.hostname, port: Number(url.port || 6333) };
}

function checkPort(host, port, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host, port });
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("timeout", () => {
      socket.destroy();
      resolve(false);
    });
    socket.once("error", () => resolve(false));
  });
}

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, {
    encoding: "utf8",
    windowsHide: true,
    stdio: "pipe",
    ...opts,
  });
  return {
    ok: res.status === 0,
    status: res.status,
    stdout: res.stdout || "",
    stderr: res.stderr || "",
  };
}

function dockerReady() {
  return run("docker", ["info"], { stdio: "ignore" }).ok;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function startDockerDesktop() {
  console.log("[qdrant] daemon de Docker apagado; abriendo Docker Desktop...");
  const programFiles = process.env.ProgramFiles || "C:\\Program Files";
  const exe = path.join(programFiles, "Docker", "Docker", "Docker Desktop.exe");
  if (!fs.existsSync(exe)) {
    console.error(`[qdrant] Docker Desktop no está en: ${exe}`);
    return false;
  }
  const child = spawn(exe, [], { detached: true, stdio: "ignore", windowsHide: true });
  child.unref();

  const deadline = Date.now() + DOCKER_DESKTOP_WAIT_MS;
  while (Date.now() < deadline) {
    if (dockerReady()) return true;
    await sleep(2_000);
  }
  console.error("[qdrant] Docker Desktop no levantó el daemon a tiempo.");
  return false;
}

async function main() {
  if ((process.env.QDRANT_ENABLED || "true").toLowerCase() === "false") {
    return; // Qdrant deshabilitado a propósito: nada que asegurar.
  }

  const { host, port } = qdrantTarget();
  console.log(`[qdrant] comprobando ${host}:${port} ...`);

  if (await checkPort(host, port)) {
    console.log("[qdrant] Qdrant ya está disponible; no se requiere acción.");
    return;
  }

  console.log("[qdrant] puerto cerrado; levantando contenedor vía docker compose...");
  if (!dockerReady() && !(await startDockerDesktop())) {
    console.error(
      "[qdrant] el daemon de Docker no responde. Abre Docker Desktop manualmente y vuelve a ejecutar pnpm dev.",
    );
    process.exit(1);
  }

  const up = run("docker", ["compose", "-f", composeFile, "up", "-d", "qdrant"]);
  if (!up.ok) {
    console.error(
      `[qdrant] docker compose falló (exit ${up.status}):\n${up.stderr || up.stdout}`,
    );
    process.exit(1);
  }

  const deadline = Date.now() + CONTAINER_WAIT_MS;
  while (Date.now() < deadline) {
    if (await checkPort(host, port, 1_000)) {
      console.log(`[qdrant] Qdrant listo en ${host}:${port}.`);
      return;
    }
    await sleep(1_000);
  }

  const logs = run("docker", ["compose", "-f", composeFile, "logs", "--tail=30", "qdrant"]);
  console.error(
    "[qdrant] el contenedor no respondió a tiempo. Últimos logs:\n" +
      (logs.stdout || logs.stderr || "(sin logs)"),
  );
  process.exit(1);
}

main().catch((err) => {
  console.error("[qdrant]", err);
  process.exit(1);
});
