export enum TurnType {
  NewQuery = "new_query",
  ConfirmationEmpty = "confirmation_empty",
  ConfirmationWithNuance = "confirmation_nuance",
}

export interface TurnClassificationResult {
  turn: TurnType;
  confidence: number;
}
