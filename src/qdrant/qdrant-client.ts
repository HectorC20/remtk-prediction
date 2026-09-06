/**
 * QdrantClient: cliente HTTP mínimo hacia Qdrant (BM25 sparse), migrado de
 * QdrantMemoryService (src) y del QdrantClient Go. Replica la tokenización
 * client-side con hash djb2 para que los índices coincidan con los ya
 * existentes en Qdrant.
 */
import type { AppConfig } from "../config";
import { warn } from "../logger";
import { createHash } from "node:crypto";

export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface QdrantPoint {
  id: string | number;
  payload: Record<string, unknown>;
  /** Mapa nombre → vector (p. ej. `{ bm25_text: { indices, values } }`). */
  vector: Record<string, SparseVector>;
}

export interface SearchResult {
  id: string;
  score: number;
  payload: Record<string, unknown>;
}

const REQ_TIMEOUT_MS = 15000;
/** Reintentos ante fallos transitorios (timeout de red / Qdrant optimizando). */
const REQ_RETRIES = 2;

/** Namespace DNS de RFC 4122 (mismo que `uuidv5.DNS` del monolith). */
const DNS_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";

/**
 * UUIDv5 determinístico (RFC 4122). Replica el `uuidv5(name, uuidv5.DNS)` del
 * monolith para que los point IDs coincidan y el re-upsert actualice en vez de
 * duplicar (Qdrant solo acepta UUID o entero como point ID).
 */
export function uuidv5(name: string, namespaceHex: string = DNS_NAMESPACE): string {
  const ns = Buffer.from(namespaceHex.replace(/-/g, ""), "hex");
  const hash = createHash("sha1").update(Buffer.concat([ns, Buffer.from(name)])).digest();
  hash[6] = (hash[6]! & 0x0f) | 0x50;
  hash[8] = (hash[8]! & 0x3f) | 0x80;
  const hex = [...hash.subarray(0, 16)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export class QdrantClient {
  private readonly base: string;
  private readonly apiKey: string;

  constructor(private readonly config: AppConfig) {
    this.base = config.qdrantUrl.replace(/\/+$/, "");
    this.apiKey = config.qdrantApiKey;
  }

  get enabled(): boolean {
    return this.config.qdrantEnabled;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["api-key"] = this.apiKey;
    return h;
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<{ status: number; data: unknown }> {
    let lastErr: unknown;
    for (let attempt = 0; attempt < REQ_RETRIES; attempt++) {
      try {
        const res = await fetch(`${this.base}${path}`, {
          method,
          headers: this.headers(),
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: AbortSignal.timeout(REQ_TIMEOUT_MS),
        });
        let data: unknown = null;
        const text = await res.text();
        if (text) {
          try {
            data = JSON.parse(text);
          } catch {
            data = text;
          }
        }
        return { status: res.status, data };
      } catch (err) {
        lastErr = err;
        // Reintenta solo fallos transitorios (timeout/red); los errores HTTP se
        // devuelven con status y no llegan aquí.
        if (attempt < REQ_RETRIES - 1) {
          await new Promise((r) => setTimeout(r, 300 * (attempt + 1)));
        }
      }
    }
    throw lastErr;
  }

  /** Crea la colección con sparse_vectors.bm25_text (modifier idf) si no existe (no destructivo). */
  async ensureSparseCollection(collection: string): Promise<void> {
    const { status } = await this.request("GET", `/collections/${collection}`);
    if (status === 200) return;
    const { status: putStatus, data } = await this.request("PUT", `/collections/${collection}`, {
      sparse_vectors: { bm25_text: { modifier: "idf" } },
    });
    if (putStatus >= 300) {
      throw new Error(`qdrant create collection ${collection}: ${JSON.stringify(data)}`);
    }
  }

  async upsertPoints(collection: string, points: QdrantPoint[]): Promise<void> {
    if (points.length === 0) return;
    const { status, data } = await this.request("PUT", `/collections/${collection}/points?wait=true`, {
      points,
    });
    if (status >= 300) {
      throw new Error(`qdrant upsert ${collection}: ${JSON.stringify(data)}`);
    }
  }

  /**
   * Consulta BM25 sparse (vector "bm25_text"). Equivalente a
   * QdrantMemoryService.searchBm25 y al SearchBm25 del Go.
   */
  async searchBm25(
    collection: string,
    queryText: string,
    limit: number,
    filter?: unknown,
    include?: string[],
  ): Promise<SearchResult[]> {
    if (!collection || queryText.trim() === "") return [];
    const sparse = this.textToSparseVector(queryText);
    if (sparse.indices.length === 0) return [];

    const body: Record<string, unknown> = {
      query: { nearest: sparse },
      using: "bm25_text",
      limit,
      with_payload: true,
      with_vector: false,
    };
    if (filter !== undefined) body.filter = filter;
    // payload_selector: devuelve solo los campos que se leen (reduce mucho la
    // respuesta; los payloads de tools incluyen el documento completo).
    if (include !== undefined && include.length > 0) {
      body.payload_selector = { include };
    }

    const { status, data } = await this.request(
      "POST",
      `/collections/${collection}/points/query`,
      body,
    );
    if (status >= 300) {
      throw new Error(`qdrant search ${collection}: ${JSON.stringify(data)}`);
    }
    const points = (data as { result?: { points?: { id: unknown; score: number; payload: Record<string, unknown> }[] } })
      ?.result?.points ?? [];
    return points.map((p) => ({
      id: String(p.id),
      score: p.score,
      payload: p.payload ?? {},
    }));
  }

  /** Replica shared.HashDjb2Qdrant (Go): iteración por code point, semilla 0. */
  hashToken(token: string): number {
    let h = 0;
    for (const ch of token) {
      h = ((h << 5) - h + ch.codePointAt(0)!) | 0;
    }
    return Math.abs(h);
  }

  /** Replica textToSparseVector (Go/src): lowercase, limpiar, split, len>1, frecuencia djb2. */
  textToSparseVector(text: string): SparseVector {
    const lower = text.toLowerCase();
    const tokens: string[] = [];
    let current = "";
    for (const ch of lower) {
      if (isAllowedTokenChar(ch)) {
        current += ch;
      } else {
        if (current.length > 1) tokens.push(current);
        current = "";
      }
    }
    if (current.length > 1) tokens.push(current);

    const freq = new Map<number, number>();
    for (const tok of tokens) {
      const h = this.hashToken(tok);
      freq.set(h, (freq.get(h) ?? 0) + 1);
    }
    const indices: number[] = [];
    const values: number[] = [];
    for (const [idx, count] of freq) {
      indices.push(idx);
      values.push(count);
    }
    return { indices, values };
  }

  /** Estado de conectividad + colecciones. */
  async getStatus(): Promise<{ ok: boolean; collections: string[] }> {
    try {
      const { status, data } = await this.request("GET", "/collections");
      if (status >= 300) return { ok: false, collections: [] };
      const result = (data as { result?: { collections?: { name: string }[] } })?.result;
      return {
        ok: true,
        collections: (result?.collections ?? []).map((c) => c.name),
      };
    } catch (err) {
      warn(`[qdrant] status falló: ${String((err as Error)?.message ?? err)}`);
      return { ok: false, collections: [] };
    }
  }
}

function isAllowedTokenChar(ch: string): boolean {
  if (ch >= "a" && ch <= "z") return true;
  if (ch >= "0" && ch <= "9") return true;
  return "áéíóúüñ".includes(ch);
}
