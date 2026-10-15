// Sonda de RAM/latencia sobre el servidor de predicción YA en ejecución (PID detectado por puerto).
// Mide RSS del proceso node antes/durante/después de una ráfaga de /predict con textos largos.
import { execFileSync } from "node:child_process";

const PORT = 6776;
const pid = process.env.PID_;

function rssMb() {
  const out = execFileSync(
    "powershell.exe",
    ["-NoProfile", "-Command", `[math]::Round((Get-Process -Id ${pid}).WorkingSet64/1MB)`],
    { encoding: "utf8", windowsHide: true },
  );
  return Number(out.trim());
}
function privMb() {
  const out = execFileSync(
    "powershell.exe",
    ["-NoProfile", "-Command", `[math]::Round((Get-Process -Id ${pid}).PrivateMemorySize64/1MB)`],
    { encoding: "utf8", windowsHide: true },
  );
  return Number(out.trim());
}

const filler = "cuéntame también detalle de cada paso del proceso y sus dependencias. ".repeat(20); // ~500+ tokens
const longText = `necesito enviar el resumen de ventas por correo a finanzas ${filler}`;

async function predict(i) {
  const t0 = Date.now();
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sessionId: `memprobe-${i}`,
        tenant: "memprobe-tenant",
        text: longText,
        source: "human",
        history: [{ role: "user", content: filler.slice(0, 400) }],
      }),
    });
    const j = await r.json();
    return { i, ms: Date.now() - t0, ok: r.status, tools: j?.tools?.length ?? 0 };
  } catch (e) {
    return { i, ms: Date.now() - t0, ok: 0, err: String(e.message) };
  }
}

console.log(`PID=${pid}  RSS inicial=${rssMb()} MB  Private=${privMb()} MB`);

const peak = { rss: 0 };
const timer = setInterval(() => {
  const r = rssMb();
  if (r > peak.rss) peak.rss = r;
}, 1500);

// Ráfaga: 6 consultas en paralelo (peor caso razonable de concurrencia de usuarios).
const t0 = Date.now();
const res = await Promise.all(Array.from({ length: 6 }, (_, i) => predict(i)));
const wall = Date.now() - t0;

clearInterval(timer);
console.log(`\nRáfaga de 6 /predict concurrentes: ${wall} ms de pared`);
for (const r of res) console.log(`  #${r.i}: ${r.ok} ${r.ms} ms tools=${r.tools ?? ''} ${r.err ?? ''}`);
console.log(`RSS pico durante la ráfaga = ${peak.rss} MB`);
console.log(`RSS después de la ráfaga   = ${rssMb()} MB  Private=${privMb()} MB`);
