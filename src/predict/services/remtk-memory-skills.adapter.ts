/**
 * RemtkMemorySkillsAdapter: Adaptador de integración directa para remtk-memory.
 *
 * Facilita a remtk-memory y a los mini-agentes:
 * 1. La propagación automática de `key-remtk` (tenant::agentId) y `sessionId`.
 * 2. El registro del ciclo de vida de entidades tras la ejecución de herramientas.
 * 3. La recuperación del contexto enriquecido de habilidades y anclaje de parámetros.
 * 4. El pre-llenado de argumentos para erradicar [FALTAN_DATOS] en ejecuciones de herramientas.
 */

import type {
  SessionEntityState,
  SkillDefinition,
  SkillPredictInput,
  SkillPredictResult,
} from "src/shared/interfaces/skill.interface";
import type { MemoryPredictionResult } from "src/shared/interfaces/memory.interface";

export interface RemtkMemoryAdapterConfig {
  baseUrl: string;
  keyRemtk: string;
  timeoutMs?: number;
}

export class RemtkMemorySkillsAdapter {
  private readonly baseUrl: string;
  private readonly keyRemtk: string;
  private readonly timeoutMs: number;

  constructor(config: RemtkMemoryAdapterConfig) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.keyRemtk = config.keyRemtk;
    this.timeoutMs = config.timeoutMs ?? 5000;
  }

  /** Headers estándar incluyendo key-remtk */
  private getHeaders(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "key-remtk": this.keyRemtk,
      "x-remtk-key": this.keyRemtk,
    };
  }

  /**
   * Registra una entidad creada o consultada en la sesión activa.
   */
  async trackEntity(params: {
    sessionId: string;
    entity: {
      id: string;
      type: string;
      name: string;
      slug?: string;
      state?: Partial<SessionEntityState["state"]>;
      attributes?: Record<string, unknown>;
    };
  }): Promise<SessionEntityState> {
    const res = await fetch(`${this.baseUrl}/entities/track`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        sessionId: params.sessionId,
        keyRemtk: this.keyRemtk,
        ...params.entity,
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`[RemtkMemorySkillsAdapter] trackEntity error: ${res.statusText}`);
    }
    return res.json() as Promise<SessionEntityState>;
  }

  /**
   * Predice la habilidad y resuelve parámetros/entidades de la sesión.
   */
  async predictSkill(params: {
    sessionId: string;
    text: string;
    attachments?: { fileId?: string; url?: string; mimeType?: string }[];
  }): Promise<SkillPredictResult> {
    const res = await fetch(`${this.baseUrl}/skills/predict`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        sessionId: params.sessionId,
        keyRemtk: this.keyRemtk,
        text: params.text,
        attachments: params.attachments,
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`[RemtkMemorySkillsAdapter] predictSkill error: ${res.statusText}`);
    }
    return res.json() as Promise<SkillPredictResult>;
  }

  /**
   * Recupera la memoria contextual completa (episódica + grafo de entidades + skill).
   */
  async predictMemory(params: {
    sessionId: string;
    text: string;
    limit?: number;
  }): Promise<MemoryPredictionResult> {
    const res = await fetch(`${this.baseUrl}/memory/predict`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        sessionId: params.sessionId,
        keyRemtk: this.keyRemtk,
        text: params.text,
        limit: params.limit ?? 8,
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`[RemtkMemorySkillsAdapter] predictMemory error: ${res.statusText}`);
    }
    return res.json() as Promise<MemoryPredictionResult>;
  }

  /**
   * Enseña o refuerza una habilidad aprendida en remtk-prediction.
   */
  async learnSkill(skill: {
    id: string;
    name: string;
    description: string;
    intentSummary: string;
    tools: string[];
    parameterGrounding?: Record<string, { id: string; name: string; metadata?: Record<string, unknown> }[]>;
  }): Promise<SkillDefinition> {
    const res = await fetch(`${this.baseUrl}/skills/learn`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        keyRemtk: this.keyRemtk,
        ...skill,
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`[RemtkMemorySkillsAdapter] learnSkill error: ${res.statusText}`);
    }
    return res.json() as Promise<SkillDefinition>;
  }

  /**
   * Limpia las entidades de la sesión.
   */
  async clearSession(sessionId: string): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/entities/clear`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        sessionId,
        keyRemtk: this.keyRemtk,
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    return res.ok;
  }
}
