import { createRequire } from "node:module";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import type { OnnxModuleLike, OnnxSessionLike, TokenizerLike, TokenizerModuleLike } from "../src/shared/interfaces";

const nodeRequire = createRequire(__filename);

async function runTest() {
  const dir = join(process.cwd(), "models", "mdeberta-v3-base-xnli-int8-onnx");
  if (!existsSync(dir)) {
    console.error("Directorio no encontrado:", dir);
    return;
  }

  const onnx: OnnxModuleLike = nodeRequire("onnxruntime-node");
  const tokenizerModule: TokenizerModuleLike = nodeRequire("@huggingface/tokenizers");

  const tokenizerJson = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8"));
  const tokenizerConfig = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8"));

  const session = await onnx.InferenceSession.create(join(dir, "model.onnx"), { executionProviders: ["cpu"] });
  const tokenizer = new tokenizerModule.Tokenizer(tokenizerJson, tokenizerConfig);

  console.log("Modelo mDeBERTa-v3-base-xnli cargado exitosamente.");

  // Las etiquetas de XNLI suelen ser: 0: entailment, 1: neutral, 2: contradiction (o viceversa según id2label)
  let id2label: Record<string, string> = { "0": "entailment", "1": "neutral", "2": "contradiction" };
  const cfgPath = join(dir, "config.json");
  if (existsSync(cfgPath)) {
    const cfg = JSON.parse(readFileSync(cfgPath, "utf8"));
    if (cfg.id2label) {
      id2label = cfg.id2label;
      console.log("id2label detectado:", id2label);
    }
  }

  const pairs = [
    {
      name: "1. Distracción Cita WhatsApp",
      premise: 'El cliente me dijo en su mensaje: "Por favor envíame un whatsapp urgente con la factura", pero tú solo agenda la reunión en el calendario para el viernes.',
      hypA: "El usuario solicita: Enviar un mensaje de WhatsApp.",
      hypB: "El usuario solicita: Crear un evento en el calendario.",
    },
    {
      name: "2. Condicional Contrafáctico SQL",
      premise: "Si tuviéramos problemas en producción te pediría ejecutar una consulta SQL, pero como todo va bien solo traduce este texto al inglés.",
      hypA: "El usuario solicita: Consultar la base de datos.",
      hypB: "El usuario solicita: Traducir un texto.",
    },
    {
      name: "3. Prohibición con Alternativa",
      premise: "Ni se te ocurra tocar la base de datos de producción; mejor busca información en la web sobre el error 500.",
      hypA: "El usuario solicita: Consultar la base de datos.",
      hypB: "El usuario solicita: Buscar información en la web.",
    },
    {
      name: "4. Doble Negación Afirmativa",
      premise: "No es que no quiera el reporte de ventas del mes, simplemente descárgalo de una vez.",
      hypA: "El usuario solicita: Descargar un reporte.",
      hypB: "El usuario solicita: Consultar la base de datos.",
    },
    {
      name: "5. Queja Casual vs Clima",
      premise: "Vaya qué calor insoportable que hace hoy por aquí en la oficina, no se puede ni trabajar.",
      hypA: "El usuario solicita: Consultar el pronóstico del clima.",
      hypB: "El usuario no solicita ninguna herramienta ni acción técnica.",
    },
  ];

  async function predictNli(premise: string, hypothesis: string) {
    const encA = tokenizer.encode(premise);
    const encB = tokenizer.encode(hypothesis);
    const ids = [...encA.ids, ...encB.ids.slice(1)].slice(0, 512);
    const len = ids.length;
    const inputIds = new BigInt64Array(len);
    const attentionMask = new BigInt64Array(len);
    for (let i = 0; i < len; i++) {
      inputIds[i] = BigInt(ids[i]);
      attentionMask[i] = 1n;
    }

    const feeds: Record<string, any> = {
      input_ids: new onnx.Tensor("int64", inputIds, [1, len]),
      attention_mask: new onnx.Tensor("int64", attentionMask, [1, len]),
    };

    const out = await session.run(feeds);
    const logits = Object.values(out)[0].data as Float32Array;

    // Softmax
    const maxLogit = Math.max(logits[0], logits[1], logits[2]);
    const exp0 = Math.exp(logits[0] - maxLogit);
    const exp1 = Math.exp(logits[1] - maxLogit);
    const exp2 = Math.exp(logits[2] - maxLogit);
    const sum = exp0 + exp1 + exp2;

    const probs = {
      [id2label["0"]]: exp0 / sum,
      [id2label["1"]]: exp1 / sum,
      [id2label["2"]]: exp2 / sum,
    };
    return probs;
  }

  for (const p of pairs) {
    console.log(`\n─────────────────────────────────────────────────────`);
    console.log(`CASO: ${p.name}`);
    console.log(`Premisa: "${p.premise}"`);
    const resA = await predictNli(p.premise, p.hypA);
    const resB = await predictNli(p.premise, p.hypB);
    console.log(`Hyp A: "${p.hypA}" ->`, Object.entries(resA).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(" | "));
    console.log(`Hyp B: "${p.hypB}" ->`, Object.entries(resB).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(" | "));
  }
}

runTest().catch(console.error);
