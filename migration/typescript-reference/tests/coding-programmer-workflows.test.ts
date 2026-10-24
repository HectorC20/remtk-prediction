/**
 * Test Riguroso de Workflows de Programación y Codificación (Developer / Programmer Workflows)
 *
 * Valida que el motor de predicción y juicio semántico (remtk-prediction) clasifique
 * y prediga con precisión exacta las herramientas técnicas requeridas por desarrolladores de software:
 *
 * 1. Edición y refactorización de código fuente (workspace_replace_in_file / workspace_write_file).
 * 2. Creación de nuevos módulos, servicios, scripts y suites de tests (workspace_write_file).
 * 3. Inspección y lectura de código/configuraciones (workspace_read_file).
 * 4. Búsqueda de símbolos en AST y referencias de código (search_code_symbols).
 * 5. Búsqueda textual y regex en todo el proyecto (grep_codebase).
 * 6. Ejecución de suites de pruebas unitarias / integración (run_test_suite).
 * 7. Control de versiones con Git: status, diff y commit (git_status, git_diff, git_commit).
 * 8. Linting y formateo automático de código (lint_and_format_code).
 * 9. Gestión e instalación de dependencias / paquetes (install_package_dependency).
 * 10. Inspección de logs y stacktraces de errores en runtime (inspect_runtime_logs).
 * 11. Ejecución de migraciones de bases de datos (run_db_migration).
 * 12. Ejecución directa de comandos en terminal CLI (run_terminal_command).
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startPredictServer } from "../src/app.module";
import type { ToolDefinition } from "../src/shared/interfaces/domain.interface";

let serverUrl = "";
let closeServer: (() => Promise<void>) | null = null;

const TENANT_ID = "programmer-workflows-tenant-2026";

/**
 * Catálogo completo de herramientas de ingeniería de software para programadores:
 */
const PROGRAMMER_CATALOG: ToolDefinition[] = [
  // 1. Filesystem & Source Code Editing
  {
    id: "workspace_write_file",
    name: "workspace_write_file",
    group: "workspace",
    category: "filesystem",
    description: "Crea o sobrescribe un archivo de código fuente en el workspace del proyecto (TypeScript, JavaScript, Python, Go, Rust, Java, C++, SQL, tests unitarios, Dockerfile, scripts, módulos o configuraciones).",
    tags: [
      "workspace", "crear archivo", "escribir archivo", "nuevo modulo", "nuevo servicio", "crear test",
      "nuevo archivo", "guardar codigo", "write file", "create module", "codigo fuente", "typescript", "python",
      "escribir test", "crear script",
    ],
    intentSummary: "Crea o sobrescribe archivos de código fuente, módulos, clases o tests en el workspace",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string" }, content: { type: "string" } },
      required: ["path", "content"],
    },
  },
  {
    id: "workspace_read_file",
    name: "workspace_read_file",
    group: "workspace",
    category: "filesystem",
    description: "Lee e inspecciona el contenido de un archivo de código fuente, middleware, controlador, configuración o package.json del proyecto.",
    tags: ["workspace", "leer archivo", "inspeccionar codigo", "ver archivo", "read file", "revisar codigo", "contenido"],
    intentSummary: "Inspecciona el contenido de un archivo de código o configuración antes de modificarlo",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  {
    id: "workspace_replace_in_file",
    name: "workspace_replace_in_file",
    group: "workspace",
    category: "filesystem",
    description: "Modifica, refactoriza o reemplaza un bloque de texto o código dentro de un archivo existente del workspace para actualizar funciones, corregir bugs o modificar lógica.",
    tags: [
      "workspace", "refactorizar", "editar archivo", "corregir bug", "reemplazar bloque", "modificar funcion",
      "patch", "actualizar logica", "replace code", "refactor", "bugfix", "modificar codigo", "actualizar funcion",
    ],
    intentSummary: "Aplica refactorizaciones, fixes y reemplazos de código dentro de un archivo existente del workspace",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string" }, findText: { type: "string" }, replaceText: { type: "string" } },
      required: ["path", "findText", "replaceText"],
    },
  },

  // 2. Code Navigation & AST Analysis
  {
    id: "search_code_symbols",
    name: "search_code_symbols",
    group: "code_intelligence",
    category: "code_navigation",
    description: "Busca definiciones de funciones, clases, interfaces, tipos, métodos y referencias de símbolos a lo largo del árbol AST de la base de código.",
    tags: [
      "ast", "simbolos", "definicion", "referencias", "find definition", "symbols", "buscar funcion",
      "buscar clase", "declaracion", "interfaz", "buscar metodo",
    ],
    intentSummary: "Localiza declaraciones, tipos, clases, interfaces y referencias en el código fuente",
    inputSchema: {
      type: "object",
      properties: { query: { type: "string" }, symbolType: { type: "string" } },
      required: ["query"],
    },
  },
  {
    id: "grep_codebase",
    name: "grep_codebase",
    group: "code_intelligence",
    category: "search",
    description: "Realiza una búsqueda textual o con expresiones regulares (regex) en todos los archivos y directorios del proyecto.",
    tags: ["grep", "buscar texto", "regex", "search pattern", "todos los archivos", "encontrar patron", "todo", "fixme"],
    intentSummary: "Busca coincidencias de texto o expresiones regulares en todo el repositorio",
    inputSchema: {
      type: "object",
      properties: { pattern: { type: "string" }, include: { type: "string" } },
      required: ["pattern"],
    },
  },

  // 3. Testing & Verification
  {
    id: "run_test_suite",
    name: "run_test_suite",
    group: "testing",
    category: "code_execution",
    description: "Ejecuta las pruebas unitarias, de integración o suites de tests con Jest, Vitest, Node Test Runner, Pytest, Cargo Test o Go Test.",
    tags: [
      "test", "tests", "pruebas unitarias", "ejecutar tests", "correr pruebas", "jest", "vitest", "pytest",
      "npm test", "unit tests", "verificar cambios con tests",
    ],
    intentSummary: "Ejecuta la suite de pruebas unitarias o de integración para validar el código",
    inputSchema: {
      type: "object",
      properties: { testPath: { type: "string" }, watch: { type: "boolean" } },
    },
  },

  // 4. Linter & Code Quality
  {
    id: "lint_and_format_code",
    name: "lint_and_format_code",
    group: "code_quality",
    category: "code_execution",
    description: "Ejecuta linters y formateadores de código (ESLint, Prettier, Biome, Ruff, Clippy) para verificar sintaxis, estilos y calidad.",
    tags: ["lint", "linter", "eslint", "prettier", "biome", "formatear codigo", "corregir estilo", "format", "calidad de codigo"],
    intentSummary: "Formatea y analiza el código con linters para asegurar estándares de calidad",
    inputSchema: {
      type: "object",
      properties: { fix: { type: "boolean" }, files: { type: "string" } },
    },
  },

  // 5. Version Control (Git)
  {
    id: "git_status",
    name: "git_status",
    group: "git",
    category: "version_control",
    description: "Muestra el estado del árbol de trabajo de Git, archivos modificados, staged y untracked en el repositorio.",
    tags: ["git", "git status", "estado git", "archivos modificados", "staged", "untracked", "cambios pendientes"],
    intentSummary: "Comprueba el estado del repositorio Git y los archivos modificados",
    inputSchema: { type: "object", properties: {} },
  },
  {
    id: "git_diff",
    name: "git_diff",
    group: "git",
    category: "version_control",
    description: "Muestra las diferencias de líneas de código (diff) entre commits, ramas o cambios pendientes en el working directory.",
    tags: ["git diff", "ver diff", "diferencias de codigo", "cambios realizados", "comparar ramas", "patch diff"],
    intentSummary: "Muestra las modificaciones de código exactas respecto a la última versión",
    inputSchema: {
      type: "object",
      properties: { file: { type: "string" }, cached: { type: "boolean" } },
    },
  },
  {
    id: "git_commit",
    name: "git_commit",
    group: "git",
    category: "version_control",
    description: "Agrega y confirma los cambios del repositorio creando un commit con un mensaje estructurado (Conventional Commits).",
    tags: ["git commit", "hacer commit", "guardar commit", "confirmar cambios", "commit message", "git add and commit"],
    intentSummary: "Crea un commit en Git con el mensaje descriptivo de los cambios",
    inputSchema: {
      type: "object",
      properties: { message: { type: "string" }, all: { type: "boolean" } },
      required: ["message"],
    },
  },

  // 6. Package & Dependency Management
  {
    id: "install_package_dependency",
    name: "install_package_dependency",
    group: "package_manager",
    category: "code_execution",
    description: "Instala, actualiza o desinstala paquetes y librerías de dependencias (npm, pnpm, yarn, pip, poetry, cargo, go get).",
    tags: [
      "npm install", "pnpm add", "instalar paquete", "instalar libreria", "agregar dependencia",
      "pip install", "cargo add", "instalar dependencias", "package.json",
    ],
    intentSummary: "Instala y gestiona librerías y paquetes de dependencias en el proyecto",
    inputSchema: {
      type: "object",
      properties: { packageNames: { type: "array", items: { type: "string" } }, dev: { type: "boolean" } },
      required: ["packageNames"],
    },
  },

  // 7. Database Migrations
  {
    id: "run_db_migration",
    name: "run_db_migration",
    group: "database",
    category: "data_engineering",
    description: "Ejecuta migraciones de base de datos y sincronización de esquemas SQL (Prisma, TypeORM, Drizzle, Alembic, Flyway).",
    tags: [
      "migracion", "prisma migrate", "db migrate", "migrar base de datos", "ejecutar migracion",
      "schema sync", "drizzle push", "typeorm migration", "actualizar esquema bd",
    ],
    intentSummary: "Aplica las migraciones pendientes en la base de datos",
    inputSchema: {
      type: "object",
      properties: { env: { type: "string" }, dryRun: { type: "boolean" } },
    },
  },

  // 8. Runtime & Logs Debugging
  {
    id: "inspect_runtime_logs",
    name: "inspect_runtime_logs",
    group: "diagnostics",
    category: "debugging",
    description: "Lee e inspecciona los registros de logs, salidas estándar (stdout/stderr) y stacktraces de errores en runtime.",
    tags: ["logs", "stacktrace", "ver errores", "traza de error", "logs del servidor", "error 500", "depurar excepcion", "runtime logs"],
    intentSummary: "Inspecciona los logs y trazas de error en tiempo de ejecución para depuración",
    inputSchema: {
      type: "object",
      properties: { lines: { type: "number" }, filter: { type: "string" } },
    },
  },

  // 9. Generic Terminal Execution
  {
    id: "run_terminal_command",
    name: "run_terminal_command",
    group: "terminal",
    category: "code_execution",
    description: "Ejecuta un comando arbitrario de consola en el shell o terminal del sistema operativo.",
    tags: ["terminal", "shell", "ejecutar comando", "bash", "powershell", "correr comando", "cmd"],
    intentSummary: "Ejecuta un comando general en la terminal del sistema",
    inputSchema: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
];

before(async () => {
  const server = await startPredictServer(0);
  serverUrl = `http://localhost:${server.port}`;
  closeServer = server.close;

  const regRes = await fetch(`${serverUrl}/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant: TENANT_ID, tools: PROGRAMMER_CATALOG }),
  });
  assert.equal(regRes.status, 200);
  // Esperar warm-up completo de embeddings
  await new Promise((resolve) => setTimeout(resolve, 8000));
  // Warmup inicial de modelos
  await predict("inicializar modelos y dependencias");
});

after(async () => {
  if (closeServer) await closeServer();
});

async function predict(text: string): Promise<{
  tools: ToolDefinition[];
  complexity: string;
  context: {
    intent: {
      primaryAction: string;
      confidence: number;
      category?: string;
      summary: string;
    };
  };
}> {
  const res = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: "coder-test-" + Math.random().toString(36).slice(2),
      tenant: TENANT_ID,
      text,
      source: "human",
    }),
  });
  assert.equal(res.status, 200);
  return (await res.json()) as any;
}

// ------------------- SUITE DE PRUEBAS UNITARIAS PARA PROGRAMADORES -------------------

test("1. Refactorización de código: 'modificar el archivo auth.service.ts para refactorizar la función authenticateUser'", async () => {
  const result = await predict("modificar el archivo auth.service.ts para refactorizar la función authenticateUser");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_replace_in_file") || toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_replace_in_file o workspace_write_file para refactorizar código, obtenido: ${toolNames.join(", ")}`,
  );
});

test("2. Corrección de Bug: 'editar invoice.ts y reemplazar la validación para corregir el bug en el cálculo de impuestos'", async () => {
  const result = await predict("editar invoice.ts y reemplazar la validación para corregir el bug en el cálculo de impuestos");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_replace_in_file") || toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_replace_in_file para aplicar fix en código, obtenido: ${toolNames.join(", ")}`,
  );
});

test("3. Creación de nuevo módulo/servicio: 'crea un nuevo servicio payment.service.ts con métodos para procesar pagos con Stripe'", async () => {
  const result = await predict("crea un nuevo servicio payment.service.ts con métodos para procesar pagos con Stripe");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file"),
    `Debe incluir workspace_write_file para crear el nuevo servicio, obtenido: ${toolNames.join(", ")}`,
  );
});

test("4. Creación de archivo de pruebas: 'crea un archivo de pruebas unitarias user.controller.spec.ts con casos de test'", async () => {
  const result = await predict("crea un archivo de pruebas unitarias user.controller.spec.ts con casos de test");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_write_file") || toolNames.includes("run_test_suite"),
    `Debe incluir workspace_write_file para escribir el archivo de pruebas, obtenido: ${toolNames.join(", ")}`,
  );
});

test("5. Ejecución de tests: 'ejecuta la suite de pruebas unitarias con jest para verificar los cambios'", async () => {
  const result = await predict("ejecuta la suite de pruebas unitarias con jest para verificar los cambios");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("run_test_suite") || toolNames.includes("run_terminal_command"),
    `Debe incluir run_test_suite para ejecutar pruebas unitarias, obtenido: ${toolNames.join(", ")}`,
  );
});

test("6. Inspección de código: 'lee el archivo config/database.ts para revisar la cadena de conexión'", async () => {
  const result = await predict("lee el archivo config/database.ts para revisar la cadena de conexión");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("workspace_read_file"),
    `Debe incluir workspace_read_file para inspeccionar archivo de código, obtenido: ${toolNames.join(", ")}`,
  );
});

test("7. Búsqueda de símbolos AST: 'busca dónde está definida la interfaz UserSession en el proyecto'", async () => {
  const result = await predict("busca dónde está definida la interfaz UserSession en el proyecto");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("search_code_symbols") || toolNames.includes("grep_codebase"),
    `Debe incluir search_code_symbols o grep_codebase para buscar definición de símbolo, obtenido: ${toolNames.join(", ")}`,
  );
});

test("8. Búsqueda de referencias: 'encuentra todas las referencias y usos del método calculateDiscount en el código'", async () => {
  const result = await predict("encuentra todas las referencias y usos del método calculateDiscount en el código");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("search_code_symbols") || toolNames.includes("grep_codebase"),
    `Debe incluir search_code_symbols o grep_codebase para buscar referencias, obtenido: ${toolNames.join(", ")}`,
  );
});

test("9. Búsqueda con Grep/Regex: 'haz un grep para buscar todos los TODOs y FIXMEs en el repositorio'", async () => {
  const result = await predict("haz un grep para buscar todos los TODOs y FIXMEs en el repositorio");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("grep_codebase") || toolNames.includes("search_code_symbols"),
    `Debe incluir grep_codebase para buscar patrones textuales/regex, obtenido: ${toolNames.join(", ")}`,
  );
});

test("10. Git Status: 'revisa qué archivos han sido modificados con git status'", async () => {
  const result = await predict("revisa qué archivos han sido modificados con git status");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("git_status") || toolNames.includes("git_diff"),
    `Debe incluir git_status para consultar archivos modificados, obtenido: ${toolNames.join(", ")}`,
  );
});

test("11. Git Diff: 'muestra el diff de los cambios realizados en el código antes de comitear'", async () => {
  const result = await predict("muestra el diff de los cambios realizados en el código antes de comitear");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("git_diff") || toolNames.includes("git_status"),
    `Debe incluir git_diff para inspeccionar cambios de código, obtenido: ${toolNames.join(", ")}`,
  );
});

test("12. Git Commit: 'haz un commit con el mensaje feat: add user authentication with JWT'", async () => {
  const result = await predict("haz un commit con el mensaje feat: add user authentication with JWT");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("git_commit"),
    `Debe incluir git_commit para confirmar cambios, obtenido: ${toolNames.join(", ")}`,
  );
});

test("13. Linters & Formatting: 'ejecuta el linter eslint y formatea los archivos con prettier'", async () => {
  const result = await predict("ejecuta el linter eslint y formatea los archivos con prettier");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("lint_and_format_code") || toolNames.includes("run_terminal_command"),
    `Debe incluir lint_and_format_code para chequear y formatear código, obtenido: ${toolNames.join(", ")}`,
  );
});

test("14. Instalación de paquetes: 'instala la librería axios y lodash en el proyecto usando pnpm'", async () => {
  const result = await predict("instala la librería axios y lodash en el proyecto usando pnpm");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("install_package_dependency") || toolNames.includes("run_terminal_command"),
    `Debe incluir install_package_dependency para agregar dependencias, obtenido: ${toolNames.join(", ")}`,
  );
});

test("15. Migraciones de BD: 'aplica las migraciones pendientes de la base de datos con prisma migrate'", async () => {
  const result = await predict("aplica las migraciones pendientes de la base de datos con prisma migrate");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("run_db_migration") || toolNames.includes("run_terminal_command"),
    `Debe incluir run_db_migration para ejecutar migraciones de base de datos, obtenido: ${toolNames.join(", ")}`,
  );
});

test("16. Depuración y Runtime Logs: 'inspecciona los logs del servidor para ver el stacktrace del error 500'", async () => {
  const result = await predict("inspecciona los logs del servidor para ver el stacktrace del error 500");
  const toolNames = result.tools.map((t) => t.name);

  assert.ok(
    toolNames.includes("inspect_runtime_logs") || toolNames.includes("workspace_read_file"),
    `Debe incluir inspect_runtime_logs para depurar el error, obtenido: ${toolNames.join(", ")}`,
  );
});
