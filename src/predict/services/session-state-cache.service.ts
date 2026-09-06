/**
 * SessionStateCacheService: buffer de estado continuo por sesión.
 *
 * Proyecta el texto entrante (español, prefijo `query:`) y mantiene un vector de
 * estado z_t por sessionId. Si el nuevo vector no cambia de tema (coseno ≥
 * TOPIC_SHIFT_THRESHOLD), aplica una combinación convexa 70/30 con el estado
 * previo; en caso contrario resetea el estado (topicShift = true).
 */
import { EmbeddingEngineService } from "../../embedding/embedding-engine";
import { TOPIC_SHIFT_THRESHOLD } from "../../shared/constants/predict/general.predict";

const BLEND_PREV = 0.7;
const BLEND_NEW = 0.3;

export interface SessionStateResult {
  zt: Float32Array;
  topicShift: boolean;
  model: string;
}

export class SessionStateCacheService {
  private readonly states = new Map<string, Float32Array>();

  constructor(private readonly engine: EmbeddingEngineService) {}

  async resolve(sessionId: string, text: string): Promise<SessionStateResult> {
    const res = await this.engine.embedQuery(text, "small", { high: true });
    const zNew = res.embedding;
    const prev = this.states.get(sessionId);

    if (!prev) {
      this.states.set(sessionId, zNew);
      return { zt: zNew, topicShift: true, model: res.model };
    }

    const cos = EmbeddingEngineService.cosine(prev, zNew);
    if (cos < TOPIC_SHIFT_THRESHOLD) {
      this.states.set(sessionId, zNew);
      return { zt: zNew, topicShift: true, model: res.model };
    }

    const blended = new Float32Array(zNew.length);
    for (let i = 0; i < zNew.length; i++) {
      blended[i] = prev[i] * BLEND_PREV + zNew[i] * BLEND_NEW;
    }
    const zt = EmbeddingEngineService.normalizeL2(blended);
    this.states.set(sessionId, zt);
    return { zt, topicShift: false, model: res.model };
  }
}
