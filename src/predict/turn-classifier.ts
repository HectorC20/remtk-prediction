/**
 * TurnClassifier: clasifica el turno (nueva consulta / confirmación) usando el
 * modelo e5-small, migrado del turn_classifier.go del server Go.
 */
import type { EmbeddingEngine } from "../embedding/embedding-engine";
import { cosine } from "../embedding/embedding-engine";
import { log } from "../logger";

export enum TurnType {
  NewQuery = "new_query",
  ConfirmationEmpty = "confirmation_empty",
  ConfirmationWithNuance = "confirmation_nuance",
}

const CONFIRM_ARCHETYPE = "confirma y continúa con el plan previo";
const NEW_QUERY_ARCHETYPE = "nueva solicitud";
const CLASSIFICATION_MARGIN = 0.03;
const NUANCE_THRESHOLD = 0.55;

export class TurnClassifier {
  private confirmEmb?: Float32Array;
  private queryEmb?: Float32Array;

  constructor(private readonly engine: EmbeddingEngine) {}

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
    const confirmScore = cosine(input.embedding, this.confirmEmb!);
    const queryScore = cosine(input.embedding, this.queryEmb!);

    // Si la consulta no queda claramente por debajo de la confirmación,
    // se trata como consulta nueva.
    let turn: TurnType;
    if (confirmScore < queryScore + CLASSIFICATION_MARGIN) {
      turn = TurnType.NewQuery;
    } else if (confirmScore < NUANCE_THRESHOLD) {
      turn = TurnType.ConfirmationWithNuance;
    } else {
      turn = TurnType.ConfirmationEmpty;
    }
    log(
      `[classifier] turn=${turn} confirm=${confirmScore.toFixed(3)} query=${queryScore.toFixed(3)} text=${JSON.stringify(text)}`,
    );
    return turn;
  }
}
