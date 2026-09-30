/**
 * API de Modelos (puerto 6777): POST /embed + GET /models + GET /health.
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
import { EmbeddingEngineService } from "./embedding.service";
import {
  ONNXMODELSIZEGTE,
  ONNXMODELSIZELARGE,
  ONNXMODELSIZESMALL,
  type ModelSize,
} from "../shared/constants/predict/version.constants";

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

  /** GET /models → estado del modelo por defecto del pipeline (env `ONNX_MODEL_SIZE`). */
  @Get("models")
  async models(): Promise<Record<string, { model: string; dim: number; modelPath: string }>> {
    const size = this.engine.defaultSize;
    return { [size]: await this.engine.getInfo(size) };
  }

  /**
   * POST /embed
   * Body: { text: string, size?: "large" | "small" } (sin prefijo; el caller aplica query:/passage:)
   * Respuesta: { embedding: number[], model: "large"|"small"|"hash", dim, modelPath }
   *
   * `size` explícito se respeta (p. ej. remtk-memory indexa en 384 dims con "small");
   * sin `size` se usa el modelo por defecto del pipeline.
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
    const size = this.resolveSize(body?.size);
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

  /** `size` explícito válido ("small"/"large"/"gte") o el default del pipeline. */
  private resolveSize(raw: unknown): ModelSize {
    return raw === ONNXMODELSIZESMALL || raw === ONNXMODELSIZELARGE || raw === ONNXMODELSIZEGTE
      ? (raw as ModelSize)
      : this.engine.defaultSize;
  }
}
