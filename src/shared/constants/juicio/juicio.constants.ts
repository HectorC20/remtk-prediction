/**
 * Defaults y arquetipos de la capa de juicio (docs/juicio.md).
 *
 * Se aplican cuando la variable de entorno correspondiente no está definida;
 * el mapeo env → valor vive en `src/config.ts`. Los arquetipos son SEMILLAS
 * semánticas (e5 proyecta cualquier idioma al mismo espacio), no diccionarios:
 * se comparan por coseno, igual que CONFIRM/META_QUESTION del clasificador.
 */

/** Interruptor maestro de la capa de juicio (env `JUICIO_ENABLED`). */
export const juicioEnabledDefault = true;

// ── 1. NLI de estado condicionado ────────────────────────────────────────────

/** Hipótesis de ejecución afirmativa (par NLI: premisa = propuesta pendiente). */
export const NLI_HYPOTHESIS = "El usuario aprueba y autoriza proceder con la acción propuesta.";

/**
 * Arquetipos de RECHAZO de la propuesta pendiente (fallback por coseno cuando
 * no hay modelo NLI cross-encoder en `JUICIO_NLI_MODEL_PATH`). Multi-idioma.
 */
export const REJECT_ARCHETYPES = [
  "no, todavía no",
  "mejor no, ahora no",
  "no lo hagas, cancela eso",
  "ni de broma, olvídalo",
  "ahora no puedo, más tarde",
  "not yet, don't do it",
  "neither, cancel that",
  "pas encore, annule ça",
];

/** Margen mínimo entre pools rechazo/confirmación para emitir veredicto. */
export const nliMarginDefault = 0.06;

// ── 2. Abstención: prototipo __NOOP__ + energía ──────────────────────────────



/**
 * Margen del prototipo NOOP sobre el mejor score de herramienta. En la banda
 * plana de cosenos de e5-small (todo cae en 0.79-0.91) el margen debe ser
 * amplio: medido en bench, small-talk da noop=1.0 vs top=0.845 y turnos
 * legítimos anafóricos llegan a noop=0.912 vs top=0.842 → entre 0.07 y 0.155.
 */
export const noopMarginDefault = 0.1;

/** Temperatura T de la energía libre E = −T·ln Σ exp(sim/T). */
export const energyTempDefault = 0.05;

/**
 * Umbral de energía τ. DESACTIVADO por defecto (τ ≥ 0 es un no-op matemático:
 * E < 0 siempre). En e5-small la energía absoluta está dominada por −max y la
 * banda es plana: una consulta legítima con top 0.83 da E ≈ −0.86 mientras un
 * small-talk plano-alto da E ≈ −0.96 — no existe τ que las separe. Activar
 * (τ < 0, env `JUICIO_ENERGY_TAU`) solo con un modelo de mayor rango dinámico.
 */
export const energyTauDefault = 0;

// ── 3. Puerta de coherencia semántica ────────────────────────────────────────

/** β de la puerta: pendiente de la sigmoide sobre el coseno consulta↔historial. */
export const gateBetaDefault = 8;

/** γ de la puerta: desplazamiento (punto medio ≈ γ/β en la escala del coseno). */
export const gateGammaDefault = 5;

/** λ mínima para concatenar el historial al prompt (por debajo se aísla el turno). */
export const gateTextMinDefault = 0.5;

// ── 4. Cross-encoder de juicio sintáctico (provider ONNX opcional) ──────────

/** Carpeta del cross-encoder (vacío = inactivo; env `JUICIO_RERANKER_MODEL_PATH`). */
export const rerankerModelPathDefault = "";

/** Peso del logit del cross-encoder en la fusión con el score del pipeline. */
export const rerankerAlphaDefault = 0.6;

// ── 5. MaxSim de interacción tardía ──────────────────────────────────────────

/**
 * Peso del MaxSim token-level en la fusión (0 = solo pipeline, 1 = solo MaxSim).
 * Con 0.5 el ruido token-level reordena turnos anafóricos sin contexto léxico
 * (bench: save_draft superó a list_invoices en "muéstrame la última otra vez");
 * 0.3 lo deja como empuje de membresía multi-intención sin imponerse al pipeline.
 */
export const maxsimWeightDefault = 0.3;

/** Tope de tokens por texto en el MaxSim (cota de cómputo O(m·n)). */
export const maxsimMaxTokensDefault = 48;

// ── 6. Especificidad (penalización por difusión semántica) ──────────────────

/** ρ de la penalización: score_ajustado = score / (1 + ρ·max(0, d(t) − d̄)). */
export const specificityRhoDefault = 0.5;

/**
 * Margen de la indexación dual: el score de `problemSpace` solo reemplaza al
 * score fusionado propio si lo supera por este margen. El reemplazo es contra
 * el score de la MISMA tool, no contra el líder, así que un margen pequeño
 * deja que cualquier problemSpace que roce la consulta salte al top (bench:
 * save_draft superó a send_email con "quiero mandar el resumen por correo").
 * 0.08 exige cuasi-paráfrasis: el match verdadero de download_report queda a
 * ~0.09 sobre su score fusionado.
 */
export const problemMarginDefault = 0.08;

/** Tope de candidatas re-puntuadas por la capa de juicio tras el rerank. */
export const juicioPostTopKDefault = 10;

/**
 * Los 16 campos de la capa en la forma de `AppConfig`, para literales de
 * config en tests (spread `...juicioConfigDefaults`). `loadConfig` NO lo usa:
 * mapea cada env var sobre su constante individual.
 */
export const juicioConfigDefaults = {
  juicioEnabled: juicioEnabledDefault,
  juicioNliMargin: nliMarginDefault,
  juicioNliModelPath: "",
  juicioNoopMargin: noopMarginDefault,
  juicioEnergyTemp: energyTempDefault,
  juicioEnergyTau: energyTauDefault,
  juicioGateBeta: gateBetaDefault,
  juicioGateGamma: gateGammaDefault,
  juicioGateTextMin: gateTextMinDefault,
  juicioRerankerModelPath: rerankerModelPathDefault,
  juicioRerankerAlpha: rerankerAlphaDefault,
  juicioMaxsimWeight: maxsimWeightDefault,
  juicioMaxsimMaxTokens: maxsimMaxTokensDefault,
  juicioSpecificityRho: specificityRhoDefault,
  juicioProblemMargin: problemMarginDefault,
  juicioPostTopK: juicioPostTopKDefault,
};
