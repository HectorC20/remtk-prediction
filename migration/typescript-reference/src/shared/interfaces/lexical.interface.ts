/**
 * Contrato del perfil léxico aprendido por canal.
 *
 * Cuarta señal del ranking: un refuerzo de pesos término → herramienta,
 * particionado por `scopeKey = tenant::agentId`. El perfil de un canal nunca
 * se comparte con otro (ver docs/entrenamiento-prediccion.md §6 y §9).
 */

/** Origen de un peso: semilla del catálogo (`seed`) o refuerzo por uso (`learned`). */
export type LexicalSource = "seed" | "learned";

/** Léxico de UNA herramienta dentro del perfil de un canal. */
export interface ToolLexicon {
  /** término normalizado → peso w(t → tool). */
  terms: Map<string, number>;
  /** Procedencia de cada término (la semilla no pisa lo aprendido, §7.3). */
  sources: Map<string, LexicalSource>;
  /** sha1 de la semilla aplicada (idempotencia del re-registro, §7.4). */
  seededHash: string;
  /** Marca temporal del último decaimiento aplicado a esta tool (§6.4). */
  lastDecay: number;
}

/** Perfil léxico de un canal. Nunca se comparte entre canales. */
export interface LexicalProfile {
  scopeKey: string;
  /** Se incrementa en cada semilla aplicada. */
  version: number;
  /** toolName → léxico de esa herramienta. */
  tools: Map<string, ToolLexicon>;
  /** document frequency del término dentro del canal. */
  df: Map<string, number>;
  /** Índice invertido término → tools que lo contienen (puntuación O(Σ postings)). */
  postings: Map<string, Set<string>>;
  /** Señales observadas por feedback (guarda de cold start, §6.5). */
  events: number;
  updatedAt: number;
}

/**
 * Resultado de puntuar una consulta contra el perfil del canal.
 * `weight` es el `learnWeightEfectivo`: 0 cuando la capa está desactivada o el
 * canal aún no alcanzó `LEARN_MIN_EVENTS` (§6.5).
 */
export interface LearnedScores {
  /** toolName → score aprendido normalizado a [0,1] por el máximo de la tanda. */
  scores: Map<string, number>;
  /** Peso efectivo del término aprendido en la fusión del ranking. */
  weight: number;
  /** Términos normalizados que la consulta aportó al perfil (observabilidad). */
  terms: string[];
}

/** Resumen del perfil para `/debug` (sin exponer los mapas completos). */
export interface LexicalProfileSnapshot {
  scopeKey: string;
  version: number;
  events: number;
  tools: number;
  terms: number;
  postings: number;
  updatedAt: string;
  learnedTools: number;
}

/** Resultado de aplicar la semilla del catálogo al perfil del canal. */
export interface LexicalSeedResult {
  seeded: number;
  cached: number;
  pruned: number;
}
