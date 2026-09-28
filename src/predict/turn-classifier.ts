import { EmbeddingEngineService } from "../embedding/embedding.service";
import { CalibrationService } from "./services/calibration.service";
import { log } from "../logger";
import { META_QUESTION_THRESHOLD } from "src/shared/constants/predict";
import { TurnType } from "../shared/interfaces";
import type { JuicioEstado } from "src/shared/interfaces/juicio.interface";
import type { TurnClassificationResult } from "src/shared/interfaces";

const META_QUESTION_PROBE =
  "preguntar qué herramientas, funciones o capacidades tiene disponibles el sistema";

const CATALOG_INQUIRY_PROBE =
  "preguntar por las herramientas o funciones disponibles de un complemento o catálogo";

import type { CrossEncoderService } from "src/juicio/services/cross-encoder.service";

export type { TurnClassificationResult };

export interface TurnClassifierContext {
  hasPendingPlan?: boolean;
  prominence?: number;
  isAutonomousPivot?: boolean;
  estado?: JuicioEstado;
}

export class TurnClassifier {
  private metaEmb?: Float32Array;
  private catalogEmb?: Float32Array;

  constructor(
    private readonly engine: EmbeddingEngineService,
    private readonly crossEncoder?: CrossEncoderService,
  ) {}

  async init(): Promise<void> {
    // La arquitectura de juicio determina la intención mediante estado de sesión y variedad topológica
  }

  /**
   * Clasifica el tipo de turno mediante la arquitectura de Juicio:
   * relación de estado de sesión (propuesta previa), veredicto NLI y prominencia
   * en la variedad de herramientas (manifold geometry), sin depender de listas de palabras ni oraciones fijas.
   */
  async classifyWithConfidence(
    text: string,
    context?: TurnClassifierContext,
  ): Promise<TurnClassificationResult> {
    // 1. Sin propuesta previa en la sesión, la consulta es siempre nueva (NewQuery).
    if (!context?.hasPendingPlan) {
      log(`[classifier] sin plan pendiente en sesión → turn=new_query conf=1.00 text=${JSON.stringify(text)}`);
      return { turn: TurnType.NewQuery, confidence: 1.0 };
    }

    // 2. Si el turno es un pivote funcional autónomo que apunta a otro dominio de herramientas:
    if (context.isAutonomousPivot) {
      log(`[classifier] pivote funcional autónomo detectado → turn=new_query conf=0.95 text=${JSON.stringify(text)}`);
      return { turn: TurnType.NewQuery, confidence: 0.95 };
    }

    // 3. Veredicto NLI directo sobre la propuesta:
    if (context.estado === "confirm") {
      const isNuanced = context.prominence !== undefined && context.prominence >= 0.035;
      const turn = isNuanced ? TurnType.ConfirmationWithNuance : TurnType.ConfirmationEmpty;
      log(`[classifier] NLI confirm → turn=${turn} conf=0.98 text=${JSON.stringify(text)}`);
      return { turn, confidence: 0.98 };
    }

    // 4. Ante una propuesta pendiente sin pivote hacia otro dominio:
    // - Si el turno es una resolución breve / anafórica (<= 5 palabras, ej. "hazlo", "el primero", "intenta de nuevo", "confirma y continúa"):
    //   corresponde a una confirmación directa (ConfirmationEmpty).
    // - Si presenta especificación en el dominio activo (prominencia >= 0.030): ConfirmationWithNuance.
    // - De lo contrario, es una nueva consulta (NewQuery).
    const words = text.trim().split(/\s+/).filter(Boolean);
    const isBriefAnaphora = words.length <= 5 && text.trim().length <= 35;
    const isNuanced = context.prominence !== undefined && context.prominence >= 0.030;

    let turn: TurnType;
    let confidence: number;
    if (isBriefAnaphora) {
      turn = TurnType.ConfirmationEmpty;
      confidence = 0.96;
    } else if (isNuanced) {
      turn = TurnType.ConfirmationWithNuance;
      confidence = 0.92;
    } else {
      turn = TurnType.NewQuery;
      confidence = 0.90;
    }

    log(
      `[classifier] juicio contextual: hasPlan=true isPivot=false anaphora=${isBriefAnaphora} prom=${context.prominence?.toFixed(3) ?? "none"} → turn=${turn} conf=${confidence.toFixed(2)} text=${JSON.stringify(text)}`,
    );

    return { turn, confidence };
  }

  async classify(text: string, context?: TurnClassifierContext): Promise<TurnType> {
    const res = await this.classifyWithConfidence(text, context);
    return res.turn;
  }

  /**
   * Detecta si el texto es una "meta-pregunta" sobre capacidades.
   * Compara por coseno contra el arquetipo canónico; si supera el umbral
   * (en el texto íntegro o en el tramo conclusivo de un párrafo multi-oración),
   * se resuelve sin herramientas.
   */
  async isMetaQuestion(text: string): Promise<boolean> {
    await this.ensureMetaEmb();
    if (!this.metaEmb) return false;
    if (await this.matchesMetaPool(text)) return true;

    const spans = text
      .split(/[.!\n]+/)
      .map((s) => s.trim())
      .filter((s) => s.length >= 15);
    if (spans.length >= 2) {
      const tail = spans[spans.length - 1];
      if (tail !== text && (await this.matchesMetaPool(tail))) return true;
    }
    return false;
  }

  private async matchesMetaPool(segment: string): Promise<boolean> {
    const input = await this.engine.embedQuery(segment, "small", { high: true });
    const s = EmbeddingEngineService.cosine(input.embedding, this.metaEmb!);
    return s >= META_QUESTION_THRESHOLD;
  }

  private async ensureMetaEmb(): Promise<void> {
    if (this.metaEmb) return;
    const res = await this.engine.embedQuery(META_QUESTION_PROBE, "small");
    this.metaEmb = res.embedding;
  }

  /**
   * Detecta semánticamente si una consulta pide el catálogo o inventario de herramientas.
   * Utiliza similitud vectorial e5-small sin requerir regex ni listas de palabras.
   */
  async isCatalogInquiry(text: string): Promise<boolean> {
    if (this.crossEncoder && (await this.crossEncoder.ready())) {
      const sCat = await this.crossEncoder.score(
        text,
        "listar o preguntar qué herramientas, funciones o catálogo tiene disponibles un complemento.",
      );
      const sSpec = await this.crossEncoder.score(
        text,
        "solicitar o verificar una herramienta para realizar una acción específica como crear o modificar.",
      );
      return sCat > 0.5 && sCat > sSpec;
    }
    if (!this.catalogEmb) {
      const res = await this.engine.embedQuery(CATALOG_INQUIRY_PROBE, "small");
      this.catalogEmb = res.embedding;
    }
    const input = await this.engine.embedQuery(text, "small", { high: true });
    const s = EmbeddingEngineService.cosine(input.embedding, this.catalogEmb);
    return s >= 0.92;
  }
}