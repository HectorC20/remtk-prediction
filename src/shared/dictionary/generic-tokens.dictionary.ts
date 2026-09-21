/**
 * Verbos y ruido genérico que aparecen en los nombres de herramienta
 * (`namespace_entidad_verbo`). No identifican ni la familia ni la entidad, por
 * eso se descartan al comparar la consulta contra el NOMBRE de la tool: lo que
 * discrimina entre dos herramientas es el namespace y la entidad, no el verbo.
 */
export const GENERIC_NAME_TOKENS = new Set([
  "listar", "list", "obtener", "get", "consultar", "buscar", "search", "ver",
  "crear", "create", "add", "nuevo", "new", "actualizar", "update", "editar",
  "edit", "eliminar", "delete", "borrar", "remove", "enviar", "send",
  "registrar", "register", "relacionar", "desrelacionar", "asignar", "quitar",
  "generar", "resolver", "mcp", "tool", "api", "app", "server", "plugin",
  "complemento", "servidor", "herramienta", "function", "call", "exec",
]);
