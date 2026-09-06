/**
 * Módulo Nest del API de Qdrant (puerto 6775).
 *
 * Igual que EmbeddingAppModule/PredictAppModule: el servicio se construye en
 * runtime con la config de entorno (loadConfig) y se registra con `useFactory`
 * bajo el token QdrantService. Al arrancar crea las colecciones si el cliente
 * externo está habilitado (equivalente al ensureCollections de createSystem).
 */
import { Module } from "@nestjs/common";
import { loadConfig } from "../config";
import { QdrantV1Controller } from "./qdrant.controller";
import { QdrantService } from "./qdrant-service";

@Module({
  controllers: [QdrantV1Controller],
  providers: [
    {
      provide: QdrantService,
      useFactory: async () => {
        const service = new QdrantService(loadConfig());
        if (service.client.enabled) await service.ensureCollections();
        return service;
      },
    },
  ],
})
export class QdrantAppModule {}
