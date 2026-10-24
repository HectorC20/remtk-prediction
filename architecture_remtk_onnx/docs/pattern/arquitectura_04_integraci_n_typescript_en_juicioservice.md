# Arquitectura 04: Integración en TypeScript con `onnxruntime-node`

## 1. Diseño del Servicio `MoEDecisionService` (`src/juicio/services/moe-decision.service.ts`)

Este servicio carga el archivo `models/remtk-juicio-moe-int8/model_quantized.onnx` y ejecuta las 3 etapas en una sola llamada nativa en C++ a través de `onnxruntime-node`.

```typescript
// src/juicio/services/moe-decision.service.ts
import * as ort from "onnxruntime-node";
import * as path from "path";
import * as fs from "fs";

export interface MoEDecisionInput {
  /** Tensor plano Float32Array de forma [1, qLen, 384] */
  queryTokens: Float32Array;
  qLen: number;
  /** Tensor plano Float32Array de forma [1, ctxLen, 384] */
  contextTokens: Float32Array;
  ctxLen: number;
  /** Tensor plano Float32Array de forma [numTools, toolLen, 384] */
  toolsTokens: Float32Array;
  numTools: number;
  toolLen: number;
}

export interface MoEDecisionOutput {
  /** Score calibrado [0, 1] para cada una de las K herramientas candidatas */
  rankedScores: number[];
  /** Probabilidades de estado NLI: [TRANSICION, CONTINUACION, CORRECCION, ABANDONO] */
  estadoProbs: {
    TRANSICION: number;
    CONTINUACION: number;
    CORRECCION: number;
    ABANDONO: number;
  };
  /** Estado dominante del turno */
  estadoDominante: "TRANSICION" | "CONTINUACION" | "CORRECCION" | "ABANDONO";
  /** Probabilidad [0, 1] de que ninguna herramienta deba ejecutarse */
  abstencionProb: number;
  /** Booleano de decisión de retorno nulo */
  debeAbstenerse: boolean;
  /** Score [0, 1] de cambio de tópico (para /memory/predict y puerta de contexto) */
  topicShiftScore: number;
  /** Pesos de activación de los 6 expertos para trazabilidad en GET /debug */
  activeExperts: number[];
}

export class MoEDecisionService {
  private session: ort.InferenceSession | null = null;
  private readonly embedDim = 384;
  private readonly abstencionThreshold = 0.52;

  constructor(private readonly modelsBasePath: string) {}

  async init(): Promise<void> {
    const modelFile = path.join(
      this.modelsBasePath,
      "remtk-juicio-moe-int8",
      "model_quantized.onnx"
    );

    if (!fs.existsSync(modelFile)) {
      throw new Error(`[MoEDecisionService] No se encontró el modelo en: ${modelFile}`);
    }

    this.session = await ort.InferenceSession.create(modelFile, {
      executionProviders: ["cpu"],
      intraOpNumThreads: 4,
      interOpNumThreads: 1,
      graphOptimizationLevel: "all",
    });
  }

  async evaluate(input: MoEDecisionInput): Promise<MoEDecisionOutput> {
    if (!this.session) {
      await this.init();
    }

    const queryTensor = new ort.Tensor("float32", input.queryTokens, [
      1,
      input.qLen,
      this.embedDim,
    ]);

    const contextTensor = new ort.Tensor("float32", input.contextTokens, [
      1,
      input.ctxLen,
      this.embedDim,
    ]);

    const toolsTensor = new ort.Tensor("float32", input.toolsTokens, [
      input.numTools,
      input.toolLen,
      this.embedDim,
    ]);

    const results = await this.session!.run({
      query_tokens: queryTensor,
      context_tokens: contextTensor,
      tools_tokens: toolsTensor,
    });

    const rankedScoresArr = Array.from(results.ranked_scores.data as Float32Array);
    const estadoArr = Array.from(results.estado_probs.data as Float32Array);
    const abstencionProb = (results.abstencion_prob.data as Float32Array)[0];
    const topicShiftScore = (results.topic_shift_score.data as Float32Array)[0];
    const activeExperts = Array.from(results.sparse_gates.data as Float32Array);

    const labels: Array<"TRANSICION" | "CONTINUACION" | "CORRECCION" | "ABANDONO"> = [
      "TRANSICION",
      "CONTINUACION",
      "CORRECCION",
      "ABANDONO",
    ];
    let maxIdx = 0;
    for (let i = 1; i < 4; i++) {
      if (estadoArr[i] > estadoArr[maxIdx]) maxIdx = i;
    }

    return {
      rankedScores: rankedScoresArr,
      estadoProbs: {
        TRANSICION: estadoArr[0],
        CONTINUACION: estadoArr[1],
        CORRECCION: estadoArr[2],
        ABANDONO: estadoArr[3],
      },
      estadoDominante: labels[maxIdx],
      abstencionProb,
      debeAbstenerse: abstencionProb >= this.abstencionThreshold,
      topicShiftScore,
      activeExperts,
    };
  }
}
```

---

## 2. Integración con el Umbral Adaptativo (`ONNX_ADAPTIVE_*`) en `Orchestrator`

Una vez que `MoEDecisionService.evaluate()` retorna en $\sim 4\text{ ms}$:

1. **Verificación de Retorno Nulo:**
   * Si `output.debeAbstenerse === true` o `cfg.MAX_OUTPUT_TOOLS === 0`, el `Orchestrator` retorna inmediatamente `tools: []` y registra en la traza de `/debug` el motivo `moe_abstention` junto con `output.estadoDominante`.
2. **Ordenamiento y Corte por Gap Natural:**
   * Se empareja cada herramienta candidata $k$ con su `output.rankedScores[k]` y se ordena de forma descendente.
   * Se filtra toda herramienta con score menor a `ONNX_ADAPTIVE_MIN_SCORE` ($0.80$).
   * Se recorre el ranking desde la posición `ONNX_ADAPTIVE_MIN_TOOLS` ($2$) hasta `ONNX_ADAPTIVE_MAX_TOOLS` ($30$); si la diferencia $s_i - s_{i+1} \ge \text{ONNX\_ADAPTIVE\_GAP\_THRESHOLD}$ ($0.03$), se aplica el corte adaptativo y se respeta el tope final `MAX_OUTPUT_TOOLS`.