export type VisualToolType =
  | "CREATE_SHAPE"
  | "MOVE_ELEMENT"
  | "RESIZE_ELEMENT"
  | "CHANGE_COLOR"
  | "ADD_TEXT";

export interface SpatialCoordinates {
  x: number; // Coordenada normalizada en el canvas [0.0 - 1.0]
  y: number; // Coordenada normalizada en el canvas [0.0 - 1.0]
}

export interface SpatialPredictDto {
  query: string; // ej. "hazlo de color azul" o "dibuja una caja aquí"
  cursor?: SpatialCoordinates; // posición normalizada del cursor
  selectedId?: string; // id del elemento actualmente enfocado/seleccionado
  sessionId?: string;
  tenant?: string;
}

export interface VisualToolParameters {
  shapeType?: "rectangle" | "circle";
  color?: string;
  text?: string;
  relativeX?: number; // normalizado 0.0 - 1.0
  relativeY?: number; // normalizado 0.0 - 1.0
  deltaX?: number;
  deltaY?: number;
  scale?: number;
  targetId?: string;
}

export interface VisualToolPayload {
  tool: VisualToolType;
  parameters: VisualToolParameters;
  confidence: number;
  executionTimeMs?: number;
  disambiguation?: string;
}
