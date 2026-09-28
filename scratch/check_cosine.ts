import { EmbeddingEngineService } from "../src/embedding/embedding.service";

async function main() {
  const engine = new EmbeddingEngineService();
  await engine.init();
  const probe = "preguntar por las herramientas o funciones disponibles de un complemento o catálogo";
  const probeEmb = (await engine.embedQuery(probe, "small")).embedding;

  const texts = [
    "Qué herramientas de mitumbes tienes",
    "Listar herramientas MCP disponibles de mitumbes para el usuario",
    "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com",
    "Verificar disponibilidad de herramienta para crear ítems en el catálogo",
    "crea el item La Pichanga Gastrobar, mejora su descripción",
    "PUES LA CATEGORÍA ES DE RESTAURANTE Y DE LA ZONA DE TUMBES!!!",
    "SI, USA LA HERRAMIENTA",
  ];

  for (const t of texts) {
    const emb = (await engine.embedQuery(t, "small", { high: true })).embedding;
    const sim = EmbeddingEngineService.cosine(emb, probeEmb);
    console.log(sim.toFixed(4), "->", t);
  }
}

main().catch(console.error);
