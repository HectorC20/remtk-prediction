/**
 * Módulo Nest del API de Predicción (puerto 6776).
 *
 * Igual que EmbeddingAppModule: el grafo se construye en runtime y por eso
 * PredictAppModule registra con `useFactory` el contexto que consume
 * PredictV1Controller (token PREDICT_CONTEXT). createSystem() levanta todo el
 * stack (engine → qdrant → classifier/keywords/rerank → orquestador) desde la
 * config de entorno (loadConfig) y hace warmup/ensureCollections al arrancar.
 */
import { Module } from "@nestjs/common";
import { loadConfig } from "../config";
import { createSystem } from "../main";
import { PREDICT_CONTEXT, PredictV1Controller } from "./predict.controller";

@Module({
  controllers: [PredictV1Controller],
  providers: [
    {
      provide: PREDICT_CONTEXT,
      useFactory: async () => {
        const system = await createSystem(loadConfig());
        return {
          orchestrator: system.orchestrator,
          engine: system.engine,
          debugger: system.debugger,
        };
      },
    },
  ],
})
export class PredictAppModule {}
