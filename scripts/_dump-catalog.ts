import { writeFileSync } from "node:fs";
import { TOTAL_ACTIVE_TOOLS } from "../tests/fixtures/mitumbes-real-catalog";

const out = TOTAL_ACTIVE_TOOLS.map((t) => ({
  name: t.name,
  description: (t.description ?? "").replace(/\s+/g, " ").slice(0, 480),
}));
writeFileSync("catalog-232.json", JSON.stringify(out), "utf8");
console.log(`catalog-232.json: ${out.length} tools`);
