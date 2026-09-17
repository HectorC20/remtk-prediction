/**
 * Extracción de keywords compartida por las 3 capas del pipeline.
 * - `extractQueryKeywords`: keywords salientes del prompt (frecuencia + stopwords).
 * - `toolKeywords`: keywords canónicas internas de la tool (tags + intentSummary + description).
 * El match cross-idioma NO es léxico aquí: lo hace KeywordService por embeddings.
 */
import { STOPWORDS } from "src/shared/dictionary/stopwords.dictionary";
import type { ToolDefinition } from "../shared/interfaces/domain.interface";

/** Stopwords ES/EN comunes para descartar tokens no informativos. */

/**
 * Verbos y ruido generico que aparecen en los nombres de herramienta
 * (`namespace_entidad_verbo`). No identifican ni la familia ni la entidad, por
 * eso se descartan al comparar la consulta contra el NOMBRE de la tool: lo que
 * discrimina entre dos herramientas es el namespace y la entidad, no el verbo.
 */
const GENERIC_NAME_TOKENS = new Set([
  "listar", "list", "obtener", "get", "consultar", "buscar", "search", "ver",
  "crear", "create", "add", "nuevo", "new", "actualizar", "update", "editar",
  "edit", "eliminar", "delete", "borrar", "remove", "enviar", "send",
  "registrar", "register", "relacionar", "desrelacionar", "asignar", "quitar",
  "generar", "resolver", "mcp", "tool", "api", "app", "server", "plugin",
  "complemento", "servidor", "herramienta", "function", "call", "exec",
]);

export function normalizeToken(raw: string): string {
  return raw
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}_-]+/gu, "")
    .trim();
}

/**
 * Tokeniza un identificador o un prompt para el match por NOMBRE: corta
 * camelCase, separa por cualquier caracter no alfanumerico (incluido `_`) y
 * pliega plurales ES/EN, de modo que `categorias`/`categories` y `categoria`/
 * `category` cuentan como el mismo termino.
 *
 * El plegado es simetrico (se aplica igual a la consulta y al nombre de la
 * tool), asi que no inventa coincidencias: solo evita perderlas por numero.
 */
export function matchTokens(raw: string): string[] {
  const folded = String(raw ?? "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
  const out: string[] = [];
  for (const piece of folded.split(/[^\p{L}\p{N}]+/u)) {
    if (piece.length < 3) continue;
    if (piece.length > 4 && piece.endsWith("ies")) out.push(`${piece.slice(0, -3)}y`);
    else if (piece.length > 4 && piece.endsWith("es")) {
      // Plural ES tras consonante: `lugares`→`lugar`. Se emiten las dos
      // variantes porque el singular también puede terminar en `e`
      // (`servicios`→`servicio`): la comparación es por pertenencia al
      // conjunto, así que emitir de más solo añade ruido inofensivo.
      out.push(piece.slice(0, -2));
      out.push(piece.slice(0, -1));
    } else if (piece.length > 3 && piece.endsWith("s")) out.push(piece.slice(0, -1));
    else out.push(piece);
  }
  return out;
}

/** Conjunto de tokens de match de un texto (consulta + keywords delegadas). */
export function matchTokenSet(text: string): Set<string> {
  return new Set(matchTokens(text));
}

/**
 * Tokens de IDENTIDAD del nombre de la tool: sus segmentos menos verbos y
 * ruido generico. Es la señal mas discriminante del catalogo (dos tools de la
 * misma familia comparten el namespace; dos de familias distintas no) y el
 * puente entre las palabras clave de la consulta y el nombre de la herramienta.
 */
export function toolIdentityTokens(name: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const t of matchTokens(name)) {
    if (GENERIC_NAME_TOKENS.has(t) || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

/**
 * Afinidad entre las palabras clave de la consulta y el NOMBRE de la tool:
 * fraccion de sus tokens de identidad (familia + entidad) presentes en la
 * consulta, en [0,1]. 0 cuando no hay tokens de consulta o la tool no tiene
 * identidad propia (nombre solo de verbo/ruido).
 */
export function nameAffinity(queryTokens: Set<string> | undefined, name: string): number {
  if (!queryTokens || queryTokens.size === 0) return 0;
  const identity = toolIdentityTokens(name);
  if (identity.length === 0) return 0;
  let hits = 0;
  for (const t of identity) if (queryTokens.has(t)) hits++;
  return hits / identity.length;
}

/**
 * Familia (namespace) de una herramienta: el primer token de identidad de su
 * NOMBRE (`mitumbes_item_crear` → `mitumbes`, `schedule_task` → `schedule`). Es
 * la unidad que agrupa las tools de un mismo servidor/complemento, la que
 * distingue una familia de otra cuando comparten grupo, categoria y tags.
 */
export function toolFamily(name: string): string | undefined {
  return toolIdentityTokens(name)[0];
}

/**
 * Familias NOMBRADAS en la consulta: las familias de las herramientas del pool
 * cuyo namespace aparece literalmente entre las palabras clave. Conjunto vacío
 * cuando la consulta no nombra ninguna familia del catalogo: sin ancla no hay
 * nada que discriminar y no se penaliza a nadie.
 */
export function namedFamilies(
  queryTokens: Set<string> | undefined,
  names: string[],
): Set<string> {
  const out = new Set<string>();
  if (!queryTokens || queryTokens.size === 0) return out;
  for (const name of names) {
    const family = toolFamily(name);
    if (family && queryTokens.has(family)) out.add(family);
  }
  return out;
}

/**
 * True si la herramienta pertenece a una familia DISTINTA de las nombradas en
 * la consulta. Solo se activa cuando `families` no esta vacio (la consulta
 * nombro al menos una familia del catalogo): en ese momento las herramientas de
 * otras familias no tienen que ver con esas palabras clave.
 *
 * Una tool sin identidad propia (nombre solo de verbo o ruido, p. ej.
 * `mcp_list`) nunca es ajena: no hay familia contra la que compararla.
 */
export function isForeignFamily(families: Set<string>, name: string): boolean {
  if (families.size === 0) return false;
  const family = toolFamily(name);
  return family !== undefined && !families.has(family);
}

/**
 * Herramientas nombradas EXPLICITAMENTE en las palabras clave delegadas: una
 * keyword que coincide con el NOMBRE de una herramienta del catálogo es una
 * orden directa de uso, no una pista semántica mas del texto.
 *
 * El match es por nombre EXACTO (case-insensitive) porque el score por
 * embeddings no distingue el verbo del nombre: `mitumbes_item_crear` y
 * `mitumbes_item_actualizar` comparten tokens de identidad (`mitumbes`, `item`)
 * y por tanto afinidad, así que sin este ancla la herramienta pedida puede no
 * salir o salir por detrás de sus hermanas.
 */
export function explicitToolNames(keywords: string[] | undefined, names: string[]): string[] {
  if (!keywords || keywords.length === 0) return [];
  const byLower = new Map<string, string>();
  for (const n of names) {
    const name = String(n ?? "").trim();
    if (name !== "") byLower.set(name.toLowerCase(), name);
  }
  const out: string[] = [];
  const seen = new Set<string>();
  for (const k of keywords) {
    if (typeof k !== "string") continue;
    const hit = byLower.get(k.trim().toLowerCase());
    if (!hit || seen.has(hit)) continue;
    seen.add(hit);
    out.push(hit);
  }
  return out;
}

/**
 * Extrae keywords salientes del prompt: descarta stopwords y tokens cortos,
 * ordena por frecuencia (palabras repetidas primero) y luego por longitud.
 */
export function extractQueryKeywords(prompt: string): string[] {
  const freq = new Map<string, number>();
  for (const raw of prompt.split(/[^\p{L}\p{N}_-]+/u)) {
    const w = normalizeToken(raw);
    if (w.length > 2 && !STOPWORDS.has(w)) {
      freq.set(w, (freq.get(w) ?? 0) + 1);
    }
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .map(([w]) => w);
}

/**
 * Keywords canónicas internas de la tool (nombre + tags + intentSummary +
 * description), deduplicadas.
 *
 * El NOMBRE va primero: sus tokens de identidad (familia + entidad) son los que
 * conectan las palabras clave de la consulta con la herramienta y los que
 * sobreviven a cualquier recorte posterior de la lista.
 *
 * La `description` es la fuente más rica de vocabulario de intención ("crear",
 * "listar", "eliminar") y la única señal que distingue entre tools que
 * comparten tags — caso típico de las tools de un mismo complemento, donde
 * todas heredan el nombre del complemento como tag.
 */
export function toolKeywords(tool: ToolDefinition): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const sources = [...toolIdentityTokens(tool.name ?? "")];
  for (const w of tool.tags ?? []) sources.push(w);
  for (const w of (tool.intentSummary ?? "").split(/\s+/)) sources.push(w);
  for (const w of (tool.description ?? "").split(/\s+/)) sources.push(w);
  for (const raw of sources) {
    const w = normalizeToken(raw);
    if (w.length > 2 && !STOPWORDS.has(w) && !seen.has(w)) {
      seen.add(w);
      out.push(w);
    }
  }
  return out;
}
