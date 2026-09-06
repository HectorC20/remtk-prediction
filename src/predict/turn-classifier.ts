import { CONFIRM_ARCHETYPE, NEW_QUERY_ARCHETYPE } from "src/shared/constants/messages/predict.constant";
import { EmbeddingEngineService } from "../embedding/embedding.service";
import { log } from "../logger";
import { TurnType } from "src/shared/dictionary/turn.dictionary";
const CLASSIFICATION_MARGIN = 0.03;
const NUANCE_THRESHOLD = 0.55;

export class TurnClassifier {
  private confirmEmb?: Float32Array;
  private queryEmb?: Float32Array;

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
}