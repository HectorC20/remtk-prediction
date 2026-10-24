---
name: remtk-moe-onnx-architect
description: Skill operativa para implementar, entrenar en CPU, auditar y mantener el motor Mixture of Experts (MoE) de 3 etapas en ONNX dentro de remtk-prediction usando los 10 documentos técnicos del proyecto.
---

# Skill: Cómo Trabajar con la Documentación del Proyecto `remtk-juicio-moe`

Este archivo define el **protocolo de trabajo** (tanto para desarrolladores humanos como para agentes de codificación asistida) al ejecutar cualquier tarea sobre `remtk-prediction` vinculada al nuevo modelo de toma de decisiones en ONNX.

---

## 1. Mapa de Navegación de los 10 Documentos

Nunca modifiques el código de `src/juicio/` ni los scripts de `training/` a ciegas. Según la tarea que vayas a realizar, carga en contexto **únicamente los archivos correspondientes**:

| Fase de Trabajo | Archivos que debes consultar | Objetivo concreto |
| :--- | :--- | :--- |
| **A. Auditoría de Recursos y Contratos** | `docs/plan/01-resolucion-necesidades-recursos.md`<br>`docs/arquitectura/02-presupuesto-tokens-tensores.md` | Verificar límites de tokens ($256/512/200\times 96$), consumo de RAM y compatibilidad con los puertos `6776`, `6777` y `6775`. |
| **B. Generación de Dataset y Destilación** | `docs/plan/02-estrategia-datos-destilacion.md` | Extraer casos desde `tests/*.test.ts`, generar *hard negatives* y pre-calcular tensores de `e5-small` en disco. |
| **C. Creación y Entrenamiento del Modelo en CPU** | `docs/instalacion/01-entorno-entrenamiento-cpu.md`<br>`docs/arquitectura/01-topologia-moe-tres-etapas.md`<br>`docs/arquitectura/03-implementacion-pytorch-moe.md`<br>`docs/plan/03-plan-entrenamiento-cpu.md` | Construir las 3 etapas en PyTorch, entrenar en CPU con la pérdida conjunta $\mathcal{L}_{\text{Total}}$ y exportar a `model_quantized.onnx`. |
| **D. Integración en TypeScript y Puesta en Producción** | `docs/arquitectura/04-integracion-typescript-onnx.md`<br>`docs/plan/04-migracion-validacion-produccion.md`<br>`docs/instalacion/02-despliegue-typescript-produccion.md` | Conectar `MoEDecisionService` en `src/juicio/`, eliminar `mDeBERTa` y `ms-marco-MiniLM`, y pasar `pnpm test:all`. |

---

## 2. Reglas de Oro Obligatorias (Alineadas con `AGENT.md`)

1. **Prohibición Absoluta de Parches Léxicos (`if/else` de palabras clave):**
   * Si el modelo falla en un test (por ejemplo, confunde una herramienta en `mitumbes-real-catalog.ts` o no se abstiene en `webpage-prediction-null.test.ts`), **está prohibido agregar listas negras de palabras o regex en TypeScript**.
   * **Acción correcta:** Agrega el caso fallido y $10$ variaciones sintéticas al dataset de `docs/plan/02-estrategia-datos-destilacion.md` y ejecuta $2$ épocas de re-entrenamiento rápido en CPU ($\sim 7\text{ minutos}$).
2. **Inmutabilidad del Extractor Base (`e5-small`):**
   * Nunca alteres los pesos de `multilingual-e5-small` porque romperías el contrato del Puerto `6777` y los vectores indexados en Qdrant. Todo el aprendizaje ocurre en las **3 etapas de `RemtkJuicioMoE`**.
3. **Caché de Tensores de Herramientas Intacta:**
   * En `src/predict/` y `src/juicio/`, jamás invoques `e5-small` para codificar definiciones de herramientas dentro del endpoint `POST /predict`. Los tensores $(L_t, 384)$ de las herramientas deben calcularse en `POST /tools` y leerse desde memoria RAM.

---

## 3. Flujo de Trabajo Paso a Paso (Checklist de Ejecución)

Sigue este orden secuencial para completar la transición de principio a fin:

- [ ] **Paso 1 (Entorno):** Seguir `docs/instalacion/01-entorno-entrenamiento-cpu.md` para crear `training/.venv` e instalar PyTorch CPU + Optimum ONNX.
- [ ] **Paso 2 (Datos):** Ejecutar la extracción de casos de `tests/` descrita en `docs/plan/02-estrategia-datos-destilacion.md` y generar el caché de tensores `.safetensors` usando `multilingual-e5-small`.
- [ ] **Paso 3 (Modelo y Entrenamiento):** Copiar la arquitectura de `docs/arquitectura/03-implementacion-pytorch-moe.md` en `training/architecture_moe.py` y entrenar con la pérdida multi-tarea de `docs/plan/03-plan-entrenamiento-cpu.md`.
- [ ] **Paso 4 (Exportación INT8):** Ejecutar `04_export_quantize_onnx.py` para producir `models/remtk-juicio-moe-int8/model_quantized.onnx` ($\sim 18.5\text{ MB}$).
- [ ] **Paso 5 (Integración TS):** Implementar `MoEDecisionService` siguiendo `docs/arquitectura/04-integracion-typescript-onnx.md` y conectarlo en `src/juicio/juicio.service.ts`.
- [ ] **Paso 6 (Verificación):** Ejecutar `pnpm test:all` según `docs/plan/04-migracion-validacion-produccion.md` y confirmar que la latencia en `/debug` cayó por debajo de $15\text{ ms}$.

---

## 4. Diagnóstico Rápido de Problemas Comunes

* **Síntoma 1: El Router activa siempre los mismos 2 expertos (`sparse_gates` fijo en `/debug`).**
  * *Causa:* El coeficiente $\lambda_5$ de $\mathcal{L}_{\text{BalanceMoE}}$ es muy bajo.
  * *Solución:* Subir $\lambda_5$ de $0.02$ a $0.05$ en el script de entrenamiento (`docs/plan/03-plan-entrenamiento-cpu.md`).
* **Síntoma 2: El modelo devuelve demasiadas herramientas en consultas simples.**
  * *Causa:* El margen $\gamma$ entre positivas y negativas durante el entrenamiento fue corto.
  * *Solución:* Subir el margen $\gamma$ de $0.25$ a $0.35$ en $\mathcal{L}_{\text{Margin}}$ para acentuar la caída abrupta del score y facilitar el corte de `ONNX_ADAPTIVE_GAP_THRESHOLD=0.03`.
* **Síntoma 3: Falsos positivos en saludos o texto pegado de páginas web.**
  * *Causa:* Falta de muestras negativas en la cabeza de abstención de la Etapa 3.
  * *Solución:* Aumentar el peso $\lambda_4$ de $\mathcal{L}_{\text{Abstención}}$ a $1.2$ e inyectar más fragmentos de texto informativo sin intención de herramienta dirigidos al **Experto 6**.