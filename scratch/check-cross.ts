import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-production-tools";
import { CrossEncoderService } from "../src/juicio/services/cross-encoder.service";

async function main() {
  const ce = new CrossEncoderService({ juicioRerankerModelPath: "models/ms-marco-minilm-l6-int8-onnx" } as any);
  const ready = await ce.ready();
  console.log("CrossEncoder ready:", ready);
  if (!ready) return;

  const q = "Verificar si existe una herramienta activa para crear ítems en el catálogo de mitumbes.com";
  for (const name of ["mitumbes_item_crear", "mitumbes_buscar", "mitumbes_item_listar", "mitumbes_lugar_crear"]) {
    const t = TOTAL_ACTIVE_TOOLS.find(x => x.name === name);
    if (!t) continue;
    const passage = `${t.name}: ${t.intentSummary || ""}. ${t.description}`;
    const score = await ce.score(q, passage);
    console.log(`${name}: ${score.toFixed(4)}`);
  }
}
main().catch(console.error);
