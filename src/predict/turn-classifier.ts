import {
  CONFIRM_ARCHETYPE,
  CONFIRM_ARCHETYPES,
  META_QUESTION_ARCHETYPES,
  NEW_QUERY_ARCHETYPE,
  NEW_QUERY_ARCHETYPES,
} from "src/shared/constants/messages/predict.constant";
import { EmbeddingEngineService } from "../embedding/embedding.service";
import { CalibrationService } from "./services/calibration.service";
import { log } from "../logger";
import {
  CLASSIFICATION_MARGIN,
  META_QUESTION_THRESHOLD,
  NUANCE_THRESHOLD,
} from "src/shared/constants/predict";
import { TurnType } from "src/shared/dictionary";

import type { TurnClassificationResult } from "src/shared/interfaces";

export type { TurnClassificationResult };

export class TurnClassifier {
  private confirmEmbs?: Float32Array[];
  private queryEmbs?: Float32Array[];
  private metaEmbs?: Float32Array[];

  constructor(private readonly engine: EmbeddingEngineService) {}

  async init(): Promise<void> {
    if (this.confirmEmbs && this.queryEmbs) return;

    const [confirms, queries] = await Promise.all([
      Promise.all(
        (CONFIRM_ARCHETYPES ?? [CONFIRM_ARCHETYPE]).map((text) =>
          this.engine.embedQuery(text, "small").then((r) => r.embedding),
        ),
      ),
      Promise.all(
        (NEW_QUERY_ARCHETYPES ?? [NEW_QUERY_ARCHETYPE]).map((text) =>
          this.engine.embedQuery(text, "small").then((r) => r.embedding),
        ),
      ),
    ]);

    this.confirmEmbs = confirms;
    this.queryEmbs = queries;
  }

  /**
   * Clasifica el tipo de turno y devuelve también la confianza probabilística calibrada.
   */
  async classifyWithConfidence(text: string): Promise<TurnClassificationResult> {
    await this.init();
    const input = await this.engine.embedQuery(text, "small", { high: true });

    // Máxima similitud contra el pool de confirmaciones
    let maxConfirm = 0;
    for (const emb of this.confirmEmbs!) {
      const s = EmbeddingEngineService.cosine(input.embedding, emb);
      if (s > maxConfirm) maxConfirm = s;
    }

    // Máxima similitud contra el pool de nuevas consultas
    let maxQuery = 0;
    for (const emb of this.queryEmbs!) {
      const s = EmbeddingEngineService.cosine(input.embedding, emb);
      if (s > maxQuery) maxQuery = s;
    }

    let turn: TurnType;
    if (maxConfirm < maxQuery + CLASSIFICATION_MARGIN) {
      turn = TurnType.NewQuery;
    } else if (maxConfirm < NUANCE_THRESHOLD) {
      turn = TurnType.ConfirmationWithNuance;
    } else {
      turn = TurnType.ConfirmationEmpty;
    }

    // Calibración de la probabilidad de decisión
    const rawDiff = Math.abs(maxConfirm - maxQuery);
    const confidence = CalibrationService.calibrateProbability(rawDiff + 0.5, {
      temperature: 0.25,
      bias: 0.5,
    });

    log(
      `[classifier] turn=${turn} confirm=${maxConfirm.toFixed(3)} query=${maxQuery.toFixed(3)} conf=${confidence.toFixed(2)} text=${JSON.stringify(text)}`,
    );

    return { turn, confidence };
  }

  async classify(text: string): Promise<TurnType> {
    const res = await this.classifyWithConfidence(text);
    return res.turn;
  }

  /**
   * Detecta si el texto es una "meta-pregunta" sobre capacidades.
   * Compara por coseno contra los arquetipos embebidos; si el máximo
   * supera el umbral (en el texto íntegro o en el tramo conclusivo de un
   * párrafo multi-oración), se resuelve sin herramientas.
   */
  async isMetaQuestion(text: string): Promise<boolean> {
    await this.ensureMetaEmbs();
    if (!this.metaEmbs || this.metaEmbs.length === 0) return false;
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
    let max = 0;
    for (const emb of this.metaEmbs!) {
      const s = EmbeddingEngineService.cosine(input.embedding, emb);
      if (s > max) max = s;
    }
    return max >= META_QUESTION_THRESHOLD;
  }

  private async ensureMetaEmbs(): Promise<void> {
    if (this.metaEmbs) return;
    this.metaEmbs = await Promise.all(
      META_QUESTION_ARCHETYPES.map((a) =>
        this.engine.embedQuery(a, "small").then((r) => r.embedding),
      ),
    );
  }
}