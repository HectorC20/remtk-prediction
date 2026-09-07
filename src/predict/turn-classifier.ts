import { CONFIRM_ARCHETYPE, META_QUESTION_ARCHETYPES, NEW_QUERY_ARCHETYPE } from "src/shared/constants/messages/predict.constant";
import { EmbeddingEngineService } from "../embedding/embedding.service";
import { log } from "../logger";
import { TurnType } from "src/shared/dictionary/turn.dictionary";
const CLASSIFICATION_MARGIN = 0.03;
const NUANCE_THRESHOLD = 0.55;
/** Coseno mínimo contra un arquetipo de capacidades para marcar meta-pregunta. */
const META_QUESTION_THRESHOLD = 0.75;

export class TurnClassifier {
  private confirmEmb?: Float32Array;
  private queryEmb?: Float32Array;
  private metaEmbs?: Float32Array[];

  constructor(private readonly engine: EmbeddingEngineService) {}

  async init(): Promise<void> {
    if (this.confirmEmb && this.queryEmb) return;
    const [confirm, query] = await Promise.all([
      this.engine.embedQuery(CONFIRM_ARCHETYPE, "small"),
      this.engine.embedQuery(NEW_QUERY_ARCHETYPE, "small"),
    ]);
    this.confirmEmb = confirm.embedding;
    this.queryEmb = query.embedding;
  }

  async classify(text: string): Promise<TurnType> {
    await this.init();
    const input = await this.engine.embedQuery(text, "small", { high: true });
    const confirmScore = EmbeddingEngineService.cosine(input.embedding, this.confirmEmb!);
    const queryScore = EmbeddingEngineService.cosine(input.embedding, this.queryEmb!);

    let turn: TurnType;
    if (confirmScore < queryScore + CLASSIFICATION_MARGIN) {
      turn = TurnType.NewQuery;
    } else if (confirmScore < NUANCE_THRESHOLD) {
      turn = TurnType.ConfirmationWithNuance;
    } else {
      turn = TurnType.ConfirmationEmpty;
    }
    
    // ✅ Arreglado el template string para que imprima las variables reales
    log(
      `[classifier] turn=${turn} confirm=${confirmScore.toFixed(3)} query=${queryScore.toFixed(3)} text=${JSON.stringify(text)}`,
    );
    return turn;
  }

  /**
   * Detecta si el texto es una "meta-pregunta" sobre capacidades (p. ej.
   * "dime qué herramientas tienes?"). Compara por coseno contra los arquetipos
   * embebidos (semántico, no diccionario); si el máximo supera el umbral, el
   * llamador debe resolver el turno sin herramientas.
   */
  async isMetaQuestion(text: string): Promise<boolean> {
    await this.ensureMetaEmbs();
    if (!this.metaEmbs || this.metaEmbs.length === 0) return false;
    const input = await this.engine.embedQuery(text, "small", { high: true });
    let max = 0;
    for (const emb of this.metaEmbs) {
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