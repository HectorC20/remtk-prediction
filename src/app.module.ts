/**
 * Bootstrap Nest del servicio.
 *
 * Cada listener HTTP se levanta con su propio módulo raíz autocontenido
 * (igual que en src/embedding): PredictAppModule → 6776, EmbeddingAppModule →
 * 6777, QdrantAppModule → 6775. start*Server(port) crea la app Nest, añade el
 * parser JSON con el mismo límite del listener Express original y devuelve el
 * handle {port, close} para el arranque en server.ts.
 */
import "reflect-metadata";
import express from "express";
import { NestFactory } from "@nestjs/core";
import type { INestApplication, Type } from "@nestjs/common";
import { EmbeddingAppModule } from "./embedding/embedding.module";
import { PredictAppModule } from "./predict/predict.module";
import { QdrantAppModule } from "./qdrant/qdrant.module";
import { LIMIT } from "src/shared/constants/predict/data.predict";

export interface ServerHandle {
  port: number;
  close(): Promise<void>;
}

/** Alias histórico del handle del listener de modelos (6777). */
export interface EmbedServerHandle extends ServerHandle {}

async function startHttp(module: Type, port: number): Promise<ServerHandle> {
  const app = await NestFactory.create<INestApplication>(module, {
    bodyParser: false,
    logger: false,
  });
  app.use(express.json({ limit: LIMIT }));
  await app.listen(port);
  const addr = app.getHttpServer().address() as { port: number } | null;
  return {
    port: addr?.port ?? port,
    close: () => app.close(),
  };
}

/** Arranca la app Nest de predicción (puerto 6776). */
export function startPredictServer(port: number): Promise<ServerHandle> {
  return startHttp(PredictAppModule, port);
}

/** Arranca la app Nest de modelos (puerto 6777). */
export function startEmbedServer(port: number): Promise<ServerHandle> {
  return startHttp(EmbeddingAppModule, port);
}

/** Arranca la app Nest de Qdrant (puerto 6775). */
export function startQdrantServer(port: number): Promise<ServerHandle> {
  return startHttp(QdrantAppModule, port);
}
