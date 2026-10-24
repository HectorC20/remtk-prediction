/**
 * Puertos del servicio unificado.
 *
 * El proceso escucha en 3 puertos (predicción, modelos ONNX y API Qdrant) y
 * esos mismos números son los endpoints que consumen los clientes.
 *
 * Nota: el motor Qdrant real (por defecto `localhost:6333`) NO se define aquí;
 * vive en `shared/constants/qdrant/general.qdrant.ts`.
 */

/** Puerto del API de predicción (/predict, /tools, /memory/predict, /debug). */
export const STRICTPORTPREDICT = 6776;

/** Puerto del API de modelos (POST /embed, e5-small). */
export const STRICTPORTEMBEDDING = 6777;

/** Puerto del API de Qdrant del servicio (BM25, keywording, indexado). */
export const STRICTPORTQDRANT = 6775;
