/**
 * API de Modelos (puerto 6777): POST /embed (e5-small) + GET /models + GET /health.
 *
 * Controlador Nest registrado por EmbeddingAppModule (embedding.module.ts); el motor
 * se inyecta con `useValue` vía forRoot. Sin prefijo global: mismos paths raíz que el
 * listener Express original (embed-server.ts, eliminado).
 */
import "reflect-metadata";
import {
  BadRequestException,
  Body,
  Controller,
  Get,
  HttpCode,
  Inject,
  InternalServerErrorException,
  Post,
} from "@nestjs/common";
import { EmbeddingEngineService } from "./embedding-engine";
import type { ModelSize } from "../shared/constants/predict/version.constants";

@Controller()
export class EmbeddingV1Controller {
  // @Inject explícito: evita depender de design:paramtypes (tsx/esbuild no lo emite).
  constructor(
    @Inject(EmbeddingEngineService) private readonly engine: EmbeddingEngineService,
  ) {}

  @Get("health")
  health(): { status: string } {
    return { status: "ok" };
  }

  /** GET /models → estado del modelo small (único servido en este proceso). */
  @Get("models")
  async models(): Promise<{ small: { model: string; dim: number; modelPath: string } }> {
    const small = await this.engine.getInfo("small");
    return { small };
  }

  /**
   * POST /embed
   * Body: { text: string, size?: "large" | "small" } (sin prefijo; el caller aplica query:/passage:)
   * Respuesta: { embedding: number[], model: "large"|"small"|"hash", dim, modelPath }
   */
  @Post("embed")
  @HttpCode(200)
  async embed(@Body() body: Record<string, unknown>): Promise<{
    embedding: number[];
    model: string;
    dim: number;
    modelPath: string;
  }> {
    const text = body?.text;
    if (typeof text !== "string" || text.trim() === "") {
      throw new BadRequestException({ error: "text requerido" });
    }
    const size: ModelSize | undefined = "small";
    try {
      const r = await this.engine.embed(text, size);
      return {
        embedding: Array.from(r.embedding),
        model: r.model,
        dim: r.dim,
        modelPath: r.modelPath,
      };
    } catch (err) {
      throw new InternalServerErrorException({
        error: String((err as Error)?.message ?? err),
      });
    }
  }
}
