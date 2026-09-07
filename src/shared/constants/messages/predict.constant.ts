export const CONFIRM_ARCHETYPE = "confirma y continúa con el plan previo";
export const NEW_QUERY_ARCHETYPE = "nueva solicitud";

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
