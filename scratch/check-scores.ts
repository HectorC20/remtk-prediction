import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";
import { loadConfig } from "../src/config";
import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { EspecificidadService } from "../src/juicio/services/especificidad.service";

async function main() {
  const config = loadConfig();
  const engine = new EmbeddingEngineService(config);
  const esp = new EspecificidadService(engine, config);
  esp.registerCatalog("test-scope", TOTAL_ACTIVE_TOOLS);

  for (const q of [
    "Qué herramientas de mitumbes tienes",
    "Listar herramientas MCP disponibles de mitumbes para el usuario",
    "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com",
    "crea el item La Pichanga Gastrobar..."
  ]) {
    const qEmb = (await engine.embedQuery(q, "small", { high: true })).embedding;
    const proj = await esp.projectManifold("test-scope", qEmb);
    console.log(`\nQuery: "${q.slice(0, 50)}..."`);
    console.log(`  Top tool: ${proj?.topTool}`);
    console.log(`  Top score: ${proj?.topScore.toFixed(4)}`);
    console.log(`  Prominence: ${proj?.prominence.toFixed(4)}`);
  }
}
main().catch(console.error);
