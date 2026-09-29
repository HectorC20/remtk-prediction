/**
 * Diagnóstico de los 6 rojos de `pnpm test` (no es una prueba: imprime medidas).
 *
 * Indexa los catálogos LITERALES de las suites que fallan (extraídos por texto, para
 * no importar los archivos de test y ejecutarlos) y muestra el `context.intent` que
 * realmente devuelve /predict, junto con la escala de calibración que fija ese
 * `confidence`.
 *
 *   ONNX_ENABLED=1 ONNX_MODELS_PATH=./models npx tsx scratch/diagnostico-intento.ts
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { startPredictServer } from "../src/app.module";
import { CalibrationService } from "../src/predict/services/calibration.service";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

const ROOT = resolve(__dirname, "..");

/** Recorta el literal `NOMBRE: ToolDefinition[] = [ ... ]` equilibrando corchetes. */
function leerCatalogo(archivo: string, nombre: string): ToolDefinition[] {
  const src = readFileSync(resolve(ROOT, archivo), "utf8");
  const marcador = `${nombre}: ToolDefinition[] = [`;
  const conExport = src.indexOf(marcador);
  if (conExport < 0) throw new Error(`no existe ${nombre} en ${archivo}`);
  const abre = conExport + marcador.length - 1;
  let depth = 0;
  let cierra = -1;
  for (let i = abre; i < src.length; i++) {
    if (src[i] === "[") depth++;
    else if (src[i] === "]") {
      depth--;
      if (depth === 0) {
        cierra = i;
        break;
      }
    }
  }
  const literal = src.slice(abre, cierra + 1);
  const ids = (literal.match(/\bid:\s*"/g) ?? []).length;
  // eslint-disable-next-line no-new-func
  const tools = new Function(`return ${literal};`)() as ToolDefinition[];
  if (tools.length !== ids) {
    throw new Error(`${nombre}: el recorte dio ${tools.length} herramientas y el fuente declara ${ids}`);
  }
  return tools;
}

type PredictResp = {
  tools: { name: string; group?: string; category?: string }[];
  context: { intent: { primaryAction: string; confidence: number; category?: string; summary: string } };
  rankedScores?: number[];
  calibratedScores?: number[];
};

async function predecir(url: string, cuerpo: Record<string, unknown>): Promise<PredictResp> {
  const res = await fetch(`${url}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cuerpo),
  });
  if (res.status !== 200) throw new Error(`status ${res.status}`);
  return (await res.json()) as PredictResp;
}

function mostrar(titulo: string, r: PredictResp): void {
  const intent = r.context.intent;
  console.log(
    `\n${titulo}\n` +
      `  tanda    = ${r.tools.map((t) => `${t.name}[grupo=${String(t.group)}]`).join(", ")}\n` +
      `  intent   = {category: ${String(intent.category)}, primaryAction: ${intent.primaryAction}, ` +
      `confidence: ${intent.confidence}}\n` +
      `  scores   = raw=${JSON.stringify((r.rankedScores ?? []).map((x) => Number(x.toFixed(3))))} ` +
      `cal=${JSON.stringify(r.calibratedScores ?? [])}`,
  );
}

async function main(): Promise<void> {
  const server = await startPredictServer(0);
  const url = `http://localhost:${server.port}`;

  const casos: { tenant: string; archivo: string; exportName: string; queries: { titulo: string; texto: string }[] }[] = [
    {
      tenant: "complex-system-tenant-2026",
      archivo: "tests/complex-system-plan.test.ts",
      exportName: "SYSTEM_CATALOG",
      queries: [
        { titulo: "Fase 3  espera categoría en [workspace, scheduler, own_tools]", texto: "Implementar lock distribuido con Redis SET NX EX para evitar double booking en reservas simultaneas" },
        { titulo: "Fase 8  espera categoría en [workspace, own_tools]", texto: "Crear seed con 10 negocios, 100 empleados y 10000 reservas, documentar API y crear .env.example" },
      ],
    },
    {
      tenant: "intent-test-tenant-2026",
      archivo: "tests/intent-recognition.test.ts",
      exportName: "CATALOG_TOOLS",
      queries: [
        { titulo: "Escenario 1  exige intent.confidence > 0.8", texto: "noticias de tumbes hoy" },
        { titulo: "Escenario 2  (no mide confianza, sirve de control)", texto: "quien es MILO J" },
      ],
    },
    {
      tenant: "continuous-flow-tenant-2026",
      archivo: "tests/continuous-flow-intent.test.ts",
      exportName: "AGENT_CATALOG",
      queries: [
        { titulo: "Turno 1  espera categoría en [workspace, filesystem, general]", texto: "qué documentos existe?" },
      ],
    },
  ];

  for (const c of casos) {
    const tools = leerCatalogo(c.archivo, c.exportName);
    const reg = await fetch(`${url}/tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant: c.tenant, tools }),
    });
    console.log(`\n[índice] ${c.tenant} tools=${tools.length} status=${reg.status}`);
    const ready = await fetch(`${url}/tools/ready?tenant=${encodeURIComponent(c.tenant)}`);
    console.log(`[listo] ${c.tenant} ready=${JSON.stringify(await ready.json())}`);
    const count = await fetch(`${url}/tools/count?tenant=${encodeURIComponent(c.tenant)}`);
    if (count.status === 200) console.log(`[cuenta] ${JSON.stringify(await count.json())}`);
    for (const q of c.queries) {
      mostrar(q.titulo, await predecir(url, { sessionId: `diag-${Math.random().toString(36).slice(2)}`, tenant: c.tenant, text: q.texto, source: "human" }));
    }
  }

  const temp = CalibrationService.DEFAULT_TEMP;
  const bias = CalibrationService.DEFAULT_BIAS;
  console.log(
    `\n[escala] confidence = sigmoid((score - ${bias}) / ${temp})\n` +
      `  score 0.75 -> ${CalibrationService.calibrateProbability(0.75)}\n` +
      `  score 0.80 -> ${CalibrationService.calibrateProbability(0.8)}\n` +
      `  score 0.85 -> ${CalibrationService.calibrateProbability(0.85)}\n` +
      `  score 0.90 -> ${CalibrationService.calibrateProbability(0.9)}\n` +
      `  score 0.95 -> ${CalibrationService.calibrateProbability(0.95)}\n` +
      `  score 1.00 -> ${CalibrationService.calibrateProbability(1)}\n` +
      `  score necesario para confidence 0.80 = ${(bias + temp * Math.log(0.8 / 0.2)).toFixed(3)} (ln(4)=1.386)`,
  );

  await server.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
