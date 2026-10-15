const P=6776;
async function run(label, text){
  const t0=Date.now();
  const r=await fetch(`http://127.0.0.1:${P}/predict`,{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({sessionId:"lat-"+label,tenant:"lat-tenant",text,source:"human"})});
  const j=await r.json();
  console.log(`${label}: ${Date.now()-t0} ms (status ${r.status}, tools=${j?.tools?.length??'?'})`);
}
await run("corta","envia el resumen por correo");
await run("larga512","necesito enviar el resumen de ventas por correo a finanzas "+"cuéntame también detalle de cada paso del proceso y sus dependencias. ".repeat(20));
await run("repetida","envia el resumen por correo");
