/**
 * Partición por AGENTE bajo el userId (tenant).
 *
 * Cada agente tiene su propio catálogo/índice; el chat general (agentId ausente
 * o vacío) conserva el comportamiento legacy con scope = tenant (hoy = userId).
 */

/** Normaliza un agentId opcional: undefined si no es string o si su trim() está vacío. */
export function normalizeAgentId(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const v = value.trim();
  return v === "" ? undefined : v;
}

/**
 * Clave de partición para catálogos, grafo del tenant y cachés de embeddings.
 * scopeKey = (agentId && String(agentId).trim()) ? `${tenant}::${agentId.trim()}` : tenant
 */
export function resolveScopeKey(tenant: string, agentId?: string): string {
  const a = normalizeAgentId(agentId);
  return a ? `${tenant}::${a}` : tenant;
}
