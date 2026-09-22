/**
 * SpatialPredictService: motor de predicción espacial de baja latencia para lienzos visuales.
 * Resuelve comandos interactivos (voz / texto) combinando posición de cursor y contexto anafórico.
 */
import { Injectable } from "@nestjs/common";
import { EmbeddingEngineService } from "src/embedding/embedding.service";
import { SPATIAL_COLOR_MAP, VISUAL_TOOLS_CATALOG } from "src/shared/constants/spatial";
import type {
  SpatialPredictDto,
  VisualToolPayload,
  VisualToolType,
} from "src/shared/interfaces/spatial.interface";
import { CalibrationService } from "./calibration.service";

@Injectable()
export class SpatialPredictService {
  constructor(private readonly engine?: EmbeddingEngineService) {}

  /**
   * Predice la acción visual requerida a partir del comando, cursor y selección.
   * La ruta crítica rápida resuelve patrones comunes en < 5ms.
   */
  async predict(dto: SpatialPredictDto): Promise<VisualToolPayload> {
    const startTime = Date.now();
    const rawQuery = (dto.query ?? "").trim();
    const query = rawQuery
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");
    const cursor = dto.cursor ?? { x: 0.5, y: 0.5 };
    const selectedId = dto.selectedId;

    // 1. Detección rápida de color
    const detectedColor = this.extractColor(query);

    // 2. Ruta crítica 1: Referencia anafórica de cambio de color
    // Si hay un elemento seleccionado y el comando menciona color o pintar
    if (
      selectedId &&
      (detectedColor ||
        query.includes("color") ||
        query.includes("pinta") ||
        query.includes("relleno"))
    ) {
      const color = detectedColor || "#2563eb";
      return {
        tool: "CHANGE_COLOR",
        parameters: {
          color,
          targetId: selectedId,
        },
        confidence: 0.98,
        disambiguation: `Color asignado al elemento activo (${selectedId})`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // 3. Ruta crítica 2: Creación de forma (directa con posición del cursor)
    const isCircle =
      query.includes("circulo") ||
      query.includes("circle") ||
      query.includes("redondo") ||
      query.includes("esfera");
    const isRectangle =
      query.includes("cuadro") ||
      query.includes("rectangulo") ||
      query.includes("caja") ||
      query.includes("box") ||
      query.includes("tarjeta") ||
      query.includes("card") ||
      query.includes("shape");

    const isExplicitCreationVerb =
      query.includes("crea") ||
      query.includes("dibuja") ||
      query.includes("inserta") ||
      query.includes("nuevo") ||
      query.includes("agrega") ||
      query.includes("add") ||
      query.includes("haz un") ||
      query.includes("haz una") ||
      query.includes("pon un") ||
      query.includes("pon una");

    const isShapeMention = isCircle || isRectangle;
    const isCreationIntent =
      isExplicitCreationVerb ||
      (isShapeMention && (query.includes("pon") || query.includes("haz") || !selectedId));

    if (isCreationIntent && (isShapeMention || !selectedId)) {
      return {
        tool: "CREATE_SHAPE",
        parameters: {
          shapeType: isCircle ? "circle" : "rectangle",
          color: detectedColor || "#2563eb",
          relativeX: Math.max(0, Math.min(1, cursor.x)),
          relativeY: Math.max(0, Math.min(1, cursor.y)),
        },
        confidence: isShapeMention ? 0.96 : 0.88,
        disambiguation: `Creación de ${isCircle ? "círculo" : "rectángulo"} en cursor (${cursor.x.toFixed(2)}, ${cursor.y.toFixed(2)})`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // 4. Ruta crítica 3: Movimiento relativo, puntos cardinales, esquinas y traslación
    const isMoveIntent =
      query.includes("muev") ||
      query.includes("traslad") ||
      query.includes("desplaz") ||
      query.includes("move") ||
      query.includes("corr") ||
      query.includes("baj") ||
      query.includes("sub") ||
      query.includes("pas") ||
      query.includes("llev") ||
      query.includes("trae") ||
      query.includes("acerc") ||
      query.includes("ponlo") ||
      query.includes("ubica") ||
      query.includes("coloca") ||
      query.includes("posicion") ||
      query.includes("arrastra") ||
      query.includes("acomoda") ||
      query.includes("alinea") ||
      query.includes("centr") ||
      query.includes("manda") ||
      query.includes("envia") ||
      query.includes("esquina") ||
      query.includes("centro") ||
      query.includes("medio") ||
      query.includes("tope") ||
      query.includes("fondo") ||
      ((query.includes("derecha") || query.includes("izquierda") || query.includes("arriba") || query.includes("abajo")) && !isCreationIntent);

    if (isMoveIntent) {
      const moveParams = this.parseMovement(query, cursor, selectedId);
      return {
        tool: "MOVE_ELEMENT",
        parameters: {
          deltaX: moveParams.deltaX,
          deltaY: moveParams.deltaY,
          relativeX: moveParams.relativeX,
          relativeY: moveParams.relativeY,
          targetId: moveParams.targetId,
        },
        confidence: 0.95,
        disambiguation: moveParams.disambiguation,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // 5. Ruta crítica 4: Redimensionamiento
    const isResizeIntent =
      query.includes("grande") ||
      query.includes("pequen") ||
      query.includes("chico") ||
      query.includes("agrand") ||
      query.includes("achic") ||
      query.includes("escal") ||
      query.includes("resize") ||
      query.includes("tamano") ||
      query.includes("dimension");

    if (isResizeIntent) {
      const isShrink =
        query.includes("pequen") ||
        query.includes("chico") ||
        query.includes("achic") ||
        query.includes("reduc");
      const scale = isShrink ? 0.75 : 1.35;
      return {
        tool: "RESIZE_ELEMENT",
        parameters: {
          scale,
          targetId: selectedId,
        },
        confidence: 0.92,
        disambiguation: `Escalar elemento a factor ${scale}x`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // 6. Ruta crítica 5: Texto / Etiqueta
    const isTextIntent =
      query.includes("texto") ||
      query.includes("escribe") ||
      query.includes("etiqueta") ||
      query.includes("titulo") ||
      query.includes("label") ||
      query.includes("text");

    if (isTextIntent) {
      const text = this.extractQuotedOrRemainingText(rawQuery);
      return {
        tool: "ADD_TEXT",
        parameters: {
          text: text || "Texto",
          relativeX: cursor.x,
          relativeY: cursor.y,
          color: detectedColor || "#0f172a",
          targetId: selectedId,
        },
        confidence: 0.90,
        disambiguation: `Insertar texto "${text || "Texto"}"`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // 7. Fallback: Búsqueda Semántica con Embeddings si engine está disponible
    if (this.engine) {
      try {
        const queryEmb = await this.engine.embedQuery(query, "small", { high: false });
        let bestTool: VisualToolType = "CREATE_SHAPE";
        let bestScore = -1;

        for (const tool of VISUAL_TOOLS_CATALOG) {
          const toolEmb = await this.engine.embedPassage(
            `${tool.intentSummary} ${tool.description} ${tool.tags.join(" ")}`,
            "small",
            { high: false },
          );
          const score = EmbeddingEngineService.cosine(queryEmb.embedding, toolEmb.embedding);
          if (score > bestScore) {
            bestScore = score;
            bestTool = tool.name as VisualToolType;
          }
        }

        const prob = CalibrationService.calibrateProbability(bestScore);
        return {
          tool: bestTool,
          parameters: {
            color: detectedColor || "#2563eb",
            relativeX: cursor.x,
            relativeY: cursor.y,
            targetId: selectedId,
          },
          confidence: prob,
          disambiguation: `Predicción semántica vectorial (${bestScore.toFixed(3)})`,
          executionTimeMs: Date.now() - startTime,
        };
      } catch {
        // Fallback simple ante error de embeddings
      }
    }

    // Fallback por defecto: Creación en cursor
    return {
      tool: "CREATE_SHAPE",
      parameters: {
        shapeType: "rectangle",
        color: detectedColor || "#2563eb",
        relativeX: cursor.x,
        relativeY: cursor.y,
      },
      confidence: 0.70,
      disambiguation: "Acción por defecto (Crear rectángulo en cursor)",
      executionTimeMs: Date.now() - startTime,
    };
  }

  private parseMovement(
    query: string,
    cursor: { x: number; y: number },
    selectedId?: string,
  ): {
    deltaX?: number;
    deltaY?: number;
    relativeX?: number;
    relativeY?: number;
    targetId?: string;
    disambiguation: string;
  } {
    // 1. Detección de Magnitud / Paso
    let step = 0.10; // 10% del canvas por defecto
    const pxMatch = query.match(/(\d+)\s*(?:px|pixeles|pixel)/i);
    if (pxMatch && pxMatch[1]) {
      step = Math.min(0.9, Math.max(0.01, parseInt(pxMatch[1], 10) / 760));
    } else if (
      query.includes("poco") ||
      query.includes("poquito") ||
      query.includes("ligeramente") ||
      query.includes("toque") ||
      query.includes("suave") ||
      query.includes("pelin") ||
      query.includes("micro")
    ) {
      step = 0.035;
    } else if (
      query.includes("mucho") ||
      query.includes("bastante") ||
      query.includes("lejos") ||
      query.includes("harto") ||
      query.includes("extremo") ||
      query.includes("todo")
    ) {
      step = 0.25;
    }

    // 2. Anclajes y Coordenadas Absolutas en el Lienzo
    // Centro
    if (
      query.includes("centro") ||
      query.includes("medio") ||
      query.includes("centrar") ||
      query.includes("centrado")
    ) {
      return {
        relativeX: 0.5,
        relativeY: 0.5,
        targetId: selectedId,
        disambiguation: "Mover al centro del lienzo (0.50, 0.50)",
      };
    }

    // Esquinas
    const isTop = query.includes("arriba") || query.includes("superior") || query.includes("tope");
    const isBottom = query.includes("abajo") || query.includes("inferior") || query.includes("fondo");
    const isLeft = query.includes("izquierda") || query.includes("izq");
    const isRight = query.includes("derecha") || query.includes("der");

    // Esquina superior izquierda
    if (
      (isTop && isLeft) ||
      query.includes("esquina superior izquierda") ||
      query.includes("arriba a la izquierda")
    ) {
      return {
        relativeX: 0.15,
        relativeY: 0.15,
        targetId: selectedId,
        disambiguation: "Mover a esquina superior izquierda (0.15, 0.15)",
      };
    }
    // Esquina superior derecha
    if (
      (isTop && isRight) ||
      query.includes("esquina superior derecha") ||
      query.includes("arriba a la derecha")
    ) {
      return {
        relativeX: 0.85,
        relativeY: 0.15,
        targetId: selectedId,
        disambiguation: "Mover a esquina superior derecha (0.85, 0.15)",
      };
    }
    // Esquina inferior izquierda
    if (
      (isBottom && isLeft) ||
      query.includes("esquina inferior izquierda") ||
      query.includes("abajo a la izquierda")
    ) {
      return {
        relativeX: 0.15,
        relativeY: 0.85,
        targetId: selectedId,
        disambiguation: "Mover a esquina inferior izquierda (0.15, 0.85)",
      };
    }
    // Esquina inferior derecha
    if (
      (isBottom && isRight) ||
      query.includes("esquina inferior derecha") ||
      query.includes("abajo a la derecha")
    ) {
      return {
        relativeX: 0.85,
        relativeY: 0.85,
        targetId: selectedId,
        disambiguation: "Mover a esquina inferior derecha (0.85, 0.85)",
      };
    }

    // Bordes / extremos absolutos
    if (query.includes("tope") || query.includes("arriba del todo") || query.includes("borde superior")) {
      return {
        relativeX: 0.5,
        relativeY: 0.12,
        targetId: selectedId,
        disambiguation: "Mover al tope superior (0.50, 0.12)",
      };
    }
    if (query.includes("fondo") || query.includes("abajo del todo") || query.includes("borde inferior")) {
      return {
        relativeX: 0.5,
        relativeY: 0.88,
        targetId: selectedId,
        disambiguation: "Mover al fondo inferior (0.50, 0.88)",
      };
    }
    if (query.includes("borde izquierdo") || query.includes("izquierda del todo")) {
      return {
        relativeX: 0.12,
        relativeY: 0.5,
        targetId: selectedId,
        disambiguation: "Mover al borde izquierdo (0.12, 0.50)",
      };
    }
    if (query.includes("borde derecho") || query.includes("derecha del todo")) {
      return {
        relativeX: 0.88,
        relativeY: 0.5,
        targetId: selectedId,
        disambiguation: "Mover al borde derecho (0.88, 0.50)",
      };
    }

    // Posición del cursor ("acá", "aquí", "cursor", "donde estoy")
    if (
      query.includes("aca") ||
      query.includes("aqui") ||
      query.includes("cursor") ||
      query.includes("puntero") ||
      query.includes("raton") ||
      query.includes("mouse") ||
      query.includes("traelo") ||
      query.includes("donde")
    ) {
      return {
        relativeX: cursor.x,
        relativeY: cursor.y,
        targetId: selectedId,
        disambiguation: `Mover hacia cursor (${cursor.x.toFixed(2)}, ${cursor.y.toFixed(2)})`,
      };
    }

    // 3. Movimientos relativos direccionales (Deltas)
    let deltaX = 0;
    let deltaY = 0;

    if (isRight) deltaX += step;
    if (isLeft) deltaX -= step;
    if (isTop || query.includes("sub") || query.includes("elev")) deltaY -= step;
    if (isBottom || query.includes("baj") || query.includes("descend")) deltaY += step;

    if (deltaX !== 0 || deltaY !== 0) {
      const dirDesc: string[] = [];
      if (deltaX > 0) dirDesc.push("derecha");
      if (deltaX < 0) dirDesc.push("izquierda");
      if (deltaY < 0) dirDesc.push("arriba");
      if (deltaY > 0) dirDesc.push("abajo");
      return {
        deltaX,
        deltaY,
        targetId: selectedId,
        disambiguation: `Desplazamiento relativo hacia ${dirDesc.join("-")} (delta: ${deltaX.toFixed(2)}, ${deltaY.toFixed(2)})`,
      };
    }

    // Default: mover hacia cursor
    return {
      relativeX: cursor.x,
      relativeY: cursor.y,
      targetId: selectedId,
      disambiguation: `Mover hacia cursor (${cursor.x.toFixed(2)}, ${cursor.y.toFixed(2)})`,
    };
  }

  private extractColor(text: string): string | undefined {
    for (const [name, hex] of Object.entries(SPATIAL_COLOR_MAP)) {
      const regex = new RegExp(`\\b${name}\\b`, "i");
      if (regex.test(text)) return hex;
    }
    return undefined;
  }

  private extractQuotedOrRemainingText(text: string): string {
    const quoteMatch = text.match(/["']([^"']+)["']/);
    if (quoteMatch && quoteMatch[1]) return quoteMatch[1];
    return text
      .replace(/\b(escribe|texto|etiqueta|pon|titulo|label|text|un|una)\b/gi, "")
      .trim();
  }
}
