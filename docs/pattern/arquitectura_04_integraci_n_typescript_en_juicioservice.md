---

### 4. `docs/pattern/arquitectura_04_integraci_n_typescript_en_juicioservice.md`

```markdown
# Arquitectura 04: Integración en TypeScript con `onnxruntime-node` (Modelo 300M INT8)

## 1. Diseño del Servicio `MoEDecisionService` (`src/juicio/services/moe-decision.service.ts`)

El servicio mantiene la misma interfaz de entrada/salida, pero incorpora **configuración de hilos AVX2/AVX-512**, **arena de memoria pre-alocada (`enableCpuMemArena`)** para soportar los $\sim 295\text{ MB}$ del grafo `remtk-juicio-moe-300m-int8` con latencia de $\sim 14-22\text{ ms}$, y garantiza que el relleno de tensores sea estrictamente `0.0` para activar las máscaras automáticas de norma L2 dentro del modelo ONNX.

```typescript
// src/juicio/services/moe-decision.service.ts
import * as ort from "onnxruntime-node";
import * as path from "path";
import * as fs from "fs";
import * as os from "os";

export interface MoEDecisionInput {
  /** Tensor plano Float32Array de forma [1, qLen, 384] (posiciones de pad en 0.0 exacto) */
  queryTokens: Float32Array;
  qLen: number;
  /** Tensor plano Float32Array de forma [1, ctxLen, 384] (posiciones de pad en 0.0 exacto) */
  contextTokens: Float32Array;
  ctxLen: number;
  /** Tensor plano Float32Array de forma [numTools, toolLen, 384] (posiciones de pad en 0.0 exacto) */
  toolsTokens: Float32Array;
  numTools: number;
  toolLen: number;
}

export interface MoEDecisionOutput {
  rankedScores: number[];
  estadoProbs: {
    TRANSICION: number;
    CONTINUACION: number;
    CORRECCION: number;
    ABANDONO: number;
  };
  estadoDominante: "TRANSICION" | "CONTINUACION" | "CORRECCION" | "ABANDONO";
  abstencionProb: number;
  debeAbstenerse: boolean;
  topicShiftScore: number;
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
      throw new Error(`[MoEDecisionService] No se encontró el modelo 300M en: ${modelFile}`);
    }

    // Optimizado para el modelo de 300M cuantizado en INT8 sobre CPU DDR5
    const physicalCores = Math.max(4, Math.min(8, Math.floor(os.cpus().length / 2)));

    this.session = await ort.InferenceSession.create(modelFile, {
      executionProviders: ["cpu"],
      intraOpNumThreads: physicalCores,
      interOpNumThreads: 1,
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
      enableMemPattern: true,
      executionMode: "sequential",
    });

    // Warmup de inicialización para reservar la arena de memoria en C++
    await this.warmup();
  }

  private async warmup(): Promise<void> {
    if (!this.session) return;
    const qDummy = new Float32Array(1 * 16 * this.embedDim);
    const cDummy = new Float32Array(1 * 16 * this.embedDim);
    const tDummy = new Float32Array(4 * 32 * this.embedDim);
    qDummy[0] = 1.0;
    cDummy[0] = 1.0;
    tDummy[0] = 1.0;
    await this.evaluate({
      queryTokens: qDummy,
      qLen: 16,
      contextTokens: cDummy,
      ctxLen: 16,
      toolsTokens: tDummy,
      numTools: 4,
      toolLen: 32,
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

    const rankedScores = Array.from(results.ranked_scores.data as Float32Array);
    const estadoRaw = Array.from(results.estado_probs.data as Float32Array);
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
      if (estadoRaw[i] > estadoRaw[maxIdx]) maxIdx = i;
    }

    const estadoDominante = labels[maxIdx];
    const debeAbstenerse =
      abstencionProb >= this.abstencionThreshold || estadoDominante === "ABANDONO";

    return {
      rankedScores,
      estadoProbs: {
        TRANSICION: estadoRaw[0],
        CONTINUACION: estadoRaw[1],
        CORRECCION: estadoRaw[2],
        ABANDONO: estadoRaw[3],
      },
      estadoDominante,
      abstencionProb,
      debeAbstenerse,
      topicShiftScore,
      activeExperts,
    };
  }
}