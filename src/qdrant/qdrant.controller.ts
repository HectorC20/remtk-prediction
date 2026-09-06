/**
 * API de Qdrant (puerto 6775) — controlador Nest.
 *
 * Contratos migrados de qdrant-server.ts (listener Express): /health, /status,
 * /tools/upsert, /tools/count, /tools/search, /memories/search, /tools/synonyms.
 * Registrado por QdrantAppModule, que provee el QdrantService (token = clase);
 * el @Inject explícito evita depender de design:paramtypes (tsx/esbuild).
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
  Query,
} from "@nestjs/common";
import { QdrantService } from "./qdrant-service";
import type { ToolDefinition } from "../shared/interfaces/domain.interface";

@Controller()
export class QdrantV1Controller {
  constructor(@Inject(QdrantService) private readonly service: QdrantService) {}

  @Get("health")
  health(): { status: string } {
    return { status: "ok" };
  }

  @Get("status")
  async status(): Promise<{ enabled: boolean; qdrant: { ok: boolean; collections: string[] } }> {
    return await this.service.status();
  }

  /** POST /tools/upsert {tenant, tools: ToolDefinition[]} → {indexed: number} */
  @Post("tools/upsert")
  @HttpCode(200)
  async upsertTools(@Body() body: Record<string, unknown>): Promise<{ indexed: number }> {
    const b = body ?? {};
    if (typeof b.tenant !== "string" || !Array.isArray(b.tools)) {
      throw new BadRequestException({ error: "tenant y tools[] requeridos" });
    }
    await this.service.indexTools(b.tenant, b.tools as ToolDefinition[]);
    return { indexed: this.service.countTools(b.tenant) };
  }

  /** GET /tools/count?tenant= → {count} */
  @Get("tools/count")
  toolsCount(@Query("tenant") tenant?: string): { count: number } {
    const t = typeof tenant === "string" ? tenant : "";
    return { count: t ? this.service.countTools(t) : 0 };
  }

  /** POST /tools/search {tenant, text, limit?} → {tools: {name, score}[]} */
  @Post("tools/search")
  @HttpCode(200)
  async searchTools(@Body() body: Record<string, unknown>): Promise<{ tools: unknown }> {
    const b = body ?? {};
    if (typeof b.tenant !== "string" || typeof b.text !== "string") {
      throw new BadRequestException({ error: "tenant y text requeridos" });
    }
    const limit = Number.isFinite(Number(b.limit)) ? Number(b.limit) : 20;
    try {
      const tools = await this.service.retrieveTools(b.tenant, b.text, limit);
      return { tools };
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /** POST /memories/search {userId, text, limit?} → {memories: MemoryDefinition[]} */
  @Post("memories/search")
  @HttpCode(200)
  async searchMemories(@Body() body: Record<string, unknown>): Promise<{ memories: unknown }> {
    const b = body ?? {};
    if (typeof b.userId !== "string" || typeof b.text !== "string") {
      throw new BadRequestException({ error: "userId y text requeridos" });
    }
    const limit = Number.isFinite(Number(b.limit)) ? Number(b.limit) : 8;
    try {
      const candidates = await this.service.searchMemories(b.userId, b.text, limit);
      return { memories: candidates.map((c) => c.memory) };
    } catch (err) {
      throw new InternalServerErrorException({ error: String((err as Error)?.message ?? err) });
    }
  }

  /** POST /tools/synonyms {text} → {expanded} */
  @Post("tools/synonyms")
  @HttpCode(200)
  async expandSynonyms(@Body() body: Record<string, unknown>): Promise<{ expanded: string }> {
    const b = body ?? {};
    if (typeof b.text !== "string") {
      throw new BadRequestException({ error: "text requerido" });
    }
    return { expanded: await this.service.expandSynonyms(b.text) };
  }
}
