/**
 * Partición por AGENTE bajo el userId (tenant) o mediante key-remtk.
 *
 * Formato de key-remtk / scopeKey:
 *   - `${tenant}::${agentId}` (ej: `7f4b4b79-ad05-4cfb-9213-d2e6575e503e::ee208755-530c-4e0d-a538-e44ff4e988dd`)
 *   - `${tenant}` (chat general)
 */

/** Normaliza un agentId opcional: undefined si no es string o si su trim() está vacío. */
export function normalizeAgentId(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const v = value.trim();
  return v === "" ? undefined : v;
}

/**
 * Parsea un key-remtk (o scopeKey) que puede venir como header 'key-remtk'/'x-remtk-key' o campo de payload.
 * Extrae { tenant, agentId, scopeKey }.
 */
export function parseKeyRemtk(
  keyRemtkHeader?: string,
  fallbackTenant?: string,
  fallbackAgentId?: string,
  bodyKeyRemtk?: string,
): { tenant: string; agentId?: string; scopeKey: string } {
  const rawKey = (keyRemtkHeader || bodyKeyRemtk || "").trim();
  if (rawKey !== "") {
    if (rawKey.includes("::")) {
      const parts = rawKey.split("::");
      const tenant = parts[0].trim();
      const agentId = parts.slice(1).join("::").trim() || undefined;
      return { tenant, agentId, scopeKey: agentId ? `${tenant}::${agentId}` : tenant };
    }
    const agentId = normalizeAgentId(fallbackAgentId);
    return { tenant: rawKey, agentId, scopeKey: resolveScopeKey(rawKey, agentId) };
  }

  const tenant = (fallbackTenant ?? "").trim();
  const agentId = normalizeAgentId(fallbackAgentId);
  return { tenant, agentId, scopeKey: resolveScopeKey(tenant, agentId) };
}

/**
 * Clave de partición para catálogos, grafo del tenant, habilidades y cachés de embeddings.
 * scopeKey = (agentId && String(agentId).trim()) ? `${tenant}::${agentId.trim()}` : tenant
 */
export function resolveScopeKey(tenant: string, agentId?: string): string {
  const t = (tenant ?? "").trim();
  if (t.includes("::")) {
    return t;
  }
  const a = normalizeAgentId(agentId);
  return a ? `${t}::${a}` : t;
}

