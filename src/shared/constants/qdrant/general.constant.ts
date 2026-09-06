/**
 * Defaults de conexión y colecciones del motor Qdrant externo.
 *
 * Es el motor real donde el servicio hace el recall BM25. No confundir con la
 * API Qdrant propia del servicio (puerto 6775), definida en
 * `shared/constants/ports/general.port.ts`.
 */

/** URL base del motor Qdrant. */
export const QDRANTURLDEFAULT = "http://localhost:6333";

/** Colección con las herramientas indexadas (recall BM25). */
export const TOOLSCOLLECTIONDEFAULT = "mcp_tools";

/** Colección de keywords RAKE por herramienta. */
export const KEYWORDSCOLLECTIONDEFAULT = "tool_keywords";

/** Colección de sinónimos para expandir queries. */
export const SYNONYMSCOLLECTIONDEFAULT = "query_synonyms";

/** Colección de memorias contextuales por usuario. */
export const MEMORIECOLLECTIONDEFAULTS = "contextual_memories";
export const MAX_KEYWORDS = 8;
/** Tope de resultados por búsqueda BM25 (150 era excesivo para payloads grandes). */
export const MAX_SEARCH_LIMIT = 50;
/** Tope de sinónimos añadidos a la query expandida. */
export const MAX_SYNONYM_EXPANSIONS = 8;
