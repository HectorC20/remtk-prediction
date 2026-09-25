/**
 * SessionStateCacheService: buffer de estado continuo por sesión.
 *
 * Proyecta el texto entrante (español, prefijo `query:`) y mantiene un vector de
 * estado z_t por sessionId. Si el nuevo vector no cambia de tema (coseno ≥
 * TOPIC_SHIFT_THRESHOLD), aplica una combinación convexa 70/30 con el estado
 * previo; en caso contrario resetea el estado (topicShift = true).
 */
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import {
  BLEND_NEW,
  BLEND_PREV,
  TOPIC_SHIFT_THRESHOLD,
} from "src/shared/constants/predict";
import { gateTextMinDefault } from "src/shared/constants/juicio";

import type { SessionStateResult } from "src/shared/interfaces";

export type { SessionStateResult };

export class SessionStateCacheService {
  private readonly states = new Map<string, Float32Array>();

  constructor(private readonly engine: EmbeddingEngineService) {}

  /** Estado actual de la sesión SIN mutarlo (lectura para la puerta de coherencia). */
  peek(sessionId: string): Float32Array | undefined {
    return this.states.get(sessionId);
  }

  /**
   * Con `gateLambda` (puerta de coherencia semántica, capa de juicio) la mezcla
   * es adaptativa: (1−λ)·zNew + λ·prev. Si la puerta aísla el turno
   * (λ < gateTextMin), resetea el estado latente al nuevo vector (topicShift = true).
   */
  async resolve(sessionId: string, text: string, gateLambda?: number): Promise<SessionStateResult> {
    const res = await this.engine.embedQuery(text, "small", { high: true });
    const zNew = res.embedding;
    const prev = this.states.get(sessionId);

    if (!prev) {
      this.states.set(sessionId, zNew);
      return { zt: zNew, topicShift: true, model: res.model };
    }

    if (gateLambda !== undefined && gateLambda < gateTextMinDefault) {
      this.states.set(sessionId, zNew);
      return { zt: zNew, topicShift: true, model: res.model };
    }

    const cos = EmbeddingEngineService.cosine(prev, zNew);
    if (cos < TOPIC_SHIFT_THRESHOLD) {
      this.states.set(sessionId, zNew);
      return { zt: zNew, topicShift: true, model: res.model };
    }

    const wPrev = gateLambda ?? BLEND_PREV;
    const wNew = gateLambda === undefined ? BLEND_NEW : 1 - gateLambda;
    const blended = new Float32Array(zNew.length);
    for (let i = 0; i < zNew.length; i++) {
      blended[i] = prev[i] * wPrev + zNew[i] * wNew;
    }
    const zt = EmbeddingEngineService.normalizeL2(blended);
    this.states.set(sessionId, zt);
    return { zt, topicShift: false, model: res.model };
  }
}
