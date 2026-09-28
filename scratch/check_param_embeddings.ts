import { EmbeddingEngineService } from "../src/embedding/embedding.service";
import { MITUMBES_VERBATIM_TOOLS } from "../tests/fixtures/mitumbes-production-tools";

async function main() {
  const engine = new EmbeddingEngineService({
    onnxEnabled: true,
    onnxModelsPath: "models",
    onnxModelSize: "small",
    qdrantUrl: "http://localhost:6333",
  } as any);

  const catTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_categoria_listar")!;
  const zoneTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_zona_listar")!;
  const itemTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_item_crear")!;
  const userTool = MITUMBES_VERBATIM_TOOLS.find(t => t.name === "mitumbes_usuario_listar")!;

  const catEmb = (await engine.embedPassage(`${catTool.name} ${catTool.description} ${catTool.intentSummary}`)).embedding;
  const zoneEmb = (await engine.embedPassage(`${zoneTool.name} ${zoneTool.description} ${zoneTool.intentSummary}`)).embedding;
  const itemEmb = (await engine.embedPassage(`${itemTool.name} ${itemTool.description} ${itemTool.intentSummary}`)).embedding;
  const userEmb = (await engine.embedPassage(`${userTool.name} ${userTool.description} ${userTool.intentSummary}`)).embedding;

  const testParams = ["categoryId", "zoneId", "userId"];

  for (const param of testParams) {
    const pEmb = (await engine.embedQuery(param, "small", { high: true })).embedding;
    console.log(`\nParam: ${param}`);
    console.log(`  mitumbes_categoria_listar: ${EmbeddingEngineService.cosine(pEmb, catEmb).toFixed(4)}`);
    console.log(`  mitumbes_zona_listar:      ${EmbeddingEngineService.cosine(pEmb, zoneEmb).toFixed(4)}`);
    console.log(`  mitumbes_item_crear:        ${EmbeddingEngineService.cosine(pEmb, itemEmb).toFixed(4)}`);
    console.log(`  mitumbes_usuario_listar:   ${EmbeddingEngineService.cosine(pEmb, userEmb).toFixed(4)}`);
  }
}

main().catch(console.error);
