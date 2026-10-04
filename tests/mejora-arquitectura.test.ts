/**
 * Test de la capa experimental `bench/mejora-arquitectura.ts`.
 *
 *  1. Unitarios (sin modelos): enmascarado, segmentación y roles.
 *  2. Integración (servidor vivo + ONNX): los 7 casos difíciles añadidos al
 *     bench y NO regresión sobre los casos que el pipeline actual ya acierta.
 *
 * Ejecución: pnpm tsx --test tests/mejora-arquitectura.test.ts
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import {
  aprueba,
  crearMejora,
  enmascarar,
  partirCaso,
  segmentar,
  type MejoraArquitectura,
} from "../bench/mejora-arquitectura";
import { BENCH_CASES } from "../bench/cases";
import type { RemtkBenchClient } from "../bench/remtk.client";

describe("unitarios: enmascarado y segmentación", () => {
  test("el código entre backticks no expone nombres de herramientas", () => {
    const t = enmascarar("mira `database_query('DROP')` y dime el dólar");
    assert.ok(!t.includes("database_query"));
    assert.ok(t.includes("[código]"));
  });

  test("auto-corrección se parte en 3 cláusulas con el marcador al frente", () => {
    const c = segmentar(
      "Genera una imagen para el flyer del evento... espera, no, no generes nada todavía, solo escribe un archivo con las notas de la reunión.",
    );
    assert.equal(c.length, 3, JSON.stringify(c));
    assert.match(c[1], /^espera/i);
    assert.match(c[2], /^solo/i);
  });

  test("'y luego' separa dominios en polisemia", () => {
    const c = segmentar(
      "Genera una imagen con el fondo nuevo para la web y luego revisa el tipo de cambio para el fondo de inversiones.",
    );
    assert.equal(c.length, 2, JSON.stringify(c));
  });

  test("un turno simple no se fragmenta", () => {
    assert.equal(segmentar("¿qué clima hace hoy en Lima?").length, 1);
    assert.equal(segmentar("mi jefe me pide el balance de ventas de ayer, es urgente").length, 1);
  });
});

const DIFICILES = [
  "auto-correccion-tiempo-real",
  "ironia-pragmatica",
  "secuestro-lexico-codigo",
  "doble-negacion-cruzada",
  "revocacion-post-propuesta",
  "condicional-contrafactico",
  "polisemia-dominio-cruzado",
];

describe("integración: baseline vs mejora", () => {
  let mejora: MejoraArquitectura;
  let remtk: RemtkBenchClient;
  const fila = new Map<string, { base: string[]; mej: string[] }>();

  before(async () => {
    ({ mejora, remtk } = await crearMejora());
    for (const c of BENCH_CASES) {
      const hist: typeof c.turns = [];
      let base: string[] = [];
      for (const t of c.turns) {
        if (t.role === "user") base = (await remtk.predict(`tbase-${c.id}`, t.content, [...hist])).tools;
        hist.push(t);
      }
      const { historial, texto } = partirCaso(c);
      const r = await mejora.resolver(c.id, texto, historial);
      fila.set(c.id, { base, mej: r.tools });
      console.log(`  ${c.id.padEnd(28)} base=[${base.join(",")}] mejora=[${r.tools.join(",")}]`);
    }
  });

  after(async () => {
    await remtk?.close();
  });

  for (const id of DIFICILES) {
    test(`caso difícil: ${id}`, () => {
      const c = BENCH_CASES.find((x) => x.id === id)!;
      const { mej } = fila.get(id)!;
      assert.ok(aprueba(c, mej), `esperado=[${c.expected.join(",") || "∅"}] mejora=[${mej.join(",")}]`);
    });
  }

  test("no regresión: todo lo que el baseline acierta, la mejora también", () => {
    const reg = BENCH_CASES.filter((c) => {
      const f = fila.get(c.id)!;
      return aprueba(c, f.base) && !aprueba(c, f.mej);
    }).map((c) => c.id);
    assert.deepEqual(reg, [], `regresiones: ${reg.join(", ")}`);
  });

  test("la mejora supera al baseline en el total", () => {
    const okB = BENCH_CASES.filter((c) => aprueba(c, fila.get(c.id)!.base)).length;
    const okM = BENCH_CASES.filter((c) => aprueba(c, fila.get(c.id)!.mej)).length;
    console.log(`  total baseline=${okB}/${BENCH_CASES.length} mejora=${okM}/${BENCH_CASES.length}`);
    assert.ok(okM > okB);
  });
});
