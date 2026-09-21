import type { TurnType } from "../dictionary/turn.dictionary";

export interface TurnClassificationResult {
  turn: TurnType;
  confidence: number;
}
