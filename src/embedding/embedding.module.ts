import { Module } from "@nestjs/common";
import { loadConfig } from "../config";
import { EmbeddingEngineService } from "./embedding-engine";
import { EmbeddingV1Controller } from "./embedding.controller";

@Module({
  controllers: [EmbeddingV1Controller],
  providers: [
    {
      provide: EmbeddingEngineService,
      useFactory: () => {
        // AppConfig es una interfaz (sin token DI): el motor se construye aquí
        // con la config real de entorno y se inicializa en background.
        const engine = new EmbeddingEngineService(loadConfig());
        engine.warmup(); // inicialización en background
        return engine;
      },
    },
  ],
  exports: [EmbeddingEngineService],
})
export class EmbeddingAppModule {}
