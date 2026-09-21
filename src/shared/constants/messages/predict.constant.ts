export const CONFIRM_ARCHETYPES = [
  "confirma y continúa con el plan previo",
  "sí, procede con eso",
  "de acuerdo, adelante",
  "correcto, hazlo",
  "ejecuta el plan",
  "proceder",
  "continúa",
  "dale para adelante",
];

export const NEW_QUERY_ARCHETYPES = [
  "nueva solicitud",
  "quiero hacer una consulta diferente",
  "cambia de tema",
  "otra tarea",
  "iniciar algo nuevo",
];

/** Arquetipo canónico de confirmación (primer elemento del pool). */
export const CONFIRM_ARCHETYPE = CONFIRM_ARCHETYPES[0];
/** Arquetipo canónico de nueva consulta (primer elemento del pool). */
export const NEW_QUERY_ARCHETYPE = NEW_QUERY_ARCHETYPES[0];

/**
 * Arquetipos de "meta-pregunta": el usuario pregunta por las capacidades del
 * sistema (qué herramientas/funciones tiene) en lugar de pedir una tarea.
 * Se comparan por coseno (no por diccionario) contra el embedding de la query;
 * si el máximo supera META_QUESTION_THRESHOLD, el turno se resuelve sin tools.
 */
export const META_QUESTION_ARCHETYPES = [
  "¿qué herramientas tienes disponibles?",
  "¿qué puedes hacer?",
  "¿cuáles son tus capacidades y funciones?",
];

/**
 * Arquetipos de "intención de interés": el usuario expresa que quiere algo
 * (comprar, contratar, pedir precio o más información). Igual que
 * META_QUESTION_ARCHETYPES se comparan por coseno contra el embedding del turno
 * (multi-idioma, sin regex): e5 proyecta cualquier idioma al mismo espacio
 * semántico, así que estos arquetipos actúan como semillas, no como diccionario.
 */
export const INTEREST_ARCHETYPES = [
  "quiero comprar este producto",
  "me interesa esta promoción",
  "quisiera más información para adquirirlo",
  "cuánto cuesta y cómo puedo pagarlo",
  "deseo contratar este servicio",
  "estoy interesado en esta oferta",
];
