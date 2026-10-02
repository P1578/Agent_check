# agentcheck

Verificador de traces de agentes de IA. Sin dependencias (solo stdlib).

> **Estado: prototipo (v0.1.0).** Útil para explorar ideas y como base de regresión. Todavía **no** está
> validado con traces reales ni listo para usarse como compuerta en producción. Ver
> [Problemas conocidos](#problemas-conocidos-en-corrección) y [Límites](#límites-de-diseño-no-son-bugs).

```bash
python -m agentcheck eval            # mide al verificador sobre el set de ejemplo: precisión/recall + qué NO detecta
python -m agentcheck demo            # corre un agente ReAct de ejemplo, lo verifica y luego lo "alucina"
python -m agentcheck verify t.json   # verifica un trace guardado ({"question","steps":[...]})
python -m pytest                     # 29 tests (requiere: pip install pytest)
```

Los comandos se ejecutan desde la raíz del repo. El paquete aún no está configurado para instalarse
(`pyproject.toml` no define `[build-system]`).

> **Nota sobre `verify`:** la CLI valida contra las herramientas y alias del dominio de ejemplo
> (`search_flights`, `search_kb`). Un trace con otras herramientas se reportará con `UNKNOWN_TOOL`. Para otro
> dominio, usa `agentcheck` como librería pasando tus propios `ToolSpec` (ver más abajo).

## Qué verifica (y cómo)

| Capa | Códigos | Cómo |
|---|---|---|
| Estructura | `EMPTY_TRACE`, `MAX_STEPS`, `NO_FINAL_ANSWER`, `MULTIPLE_FINALS`, `CALL_WITHOUT_RESULT`, `ORPHAN_RESULT`, `STEPS_AFTER_FINAL`, `UNKNOWN_TOOL`, `INVALID_ARGS`, `UNRECOVERED_TOOL_ERROR`, `RETRY_NO_CHANGE`, `LOOP` | Recorrido **secuencial** del trace; args validados contra `ToolSpec`. Un error de herramienta *recuperado* es advertencia; uno ignorado es error. |
| Grounding | `FABRICATED_FACT`, `MISATTRIBUTED_FACT`, `UNKNOWN_ENTITY`, `WEAK_SUPPORT`, `UNSUPPORTED_CLAIM` | Los datos de la respuesta final deben salir de lo que devolvieron las herramientas y pertenecer a la entidad correcta. **Solo se reconocen estos formatos:** montos con `$` (`$1,200` = `$1200`; `$120` ≠ `$1200`), horas (`7am` = `07:00`), fechas `YYYY-MM-DD` y pesos en `kg`. `UNSUPPORTED_CLAIM` solo aparece si pasas un `claim_verifier`. |
| Juez (opcional) | `JUDGE_LOW_SCORE`, `JUDGE_UNAVAILABLE` | Rúbrica con salida JSON validada (puntajes 1–5). `PairwiseJudge` compara en ambos órdenes para detectar sesgo de posición (es una utilidad aparte; `verify()` usa `RubricJudge`). |

**Qué cuenta como "pasa":** `report.passed` es verdadero si no hay **errores** y el reporte está completo.
Las **advertencias** (`UNKNOWN_ENTITY`, `WEAK_SUPPORT`, `RETRY_NO_CHANGE`, `LOOP`, `MULTIPLE_FINALS`,
`JUDGE_UNAVAILABLE`) **no hacen fallar** el reporte por sí solas. Revisa `report.warnings` si necesitas más rigor.

## Medición actual

`python -m agentcheck eval` corre **26 casos escritos a mano**: 22 estáticos y 4 semánticos.

| | Resultado |
|---|---|
| Estáticos (22) | 15/15 fallos detectados, 0 falsos positivos (TP=15 FN=0 FP=0 TN=7) |
| Semánticos (4) | **0/4 detectados sin juez LLM** |

Cómo leerlo: los casos los escribió una sola persona y **reutilizan los mismos datos que el agente de demo**
(`Viva $1200`, `Aeromexico $2100`, la política de equipaje). Por eso el 100 % sirve como **prueba de regresión**,
no como estimación de desempeño real. Pruebas posteriores con variantes fuera del set encontraron fallos
(ver abajo).

## Problemas conocidos (en corrección)

Detectados en una revisión del código el 2026-10-01. Marca cada punto cuando se corrija.

### Bugs

- [ ] **Un fallo de red del juez rompe `verify()` en lugar de dejar el reporte incompleto.**
  `verify()` solo captura `JudgeError`; los errores de red/HTTP de `AnthropicBackend` (`URLError`, timeouts)
  se propagan como excepción. Una salida inválida del juez sí produce `JUDGE_UNAVAILABLE` y `complete=False`;
  una caída de red, hoy, no.
- [ ] **Un `claim_verifier` que falla también lanza excepción.** La llamada a `LLMClaimVerifier` no está
  protegida: una respuesta inválida del modelo propaga `JudgeError` fuera de `verify()`.
- [ ] **Los números de distinto tipo se confunden (falso negativo).** Montos y `kg` comparten un mismo conjunto
  de números por documento. Ejemplo observado: con la política "10kg mano gratis" en la evidencia, la respuesta
  `Viva cuesta $10.` **pasa**.
- [ ] **Falso positivo con siglas de moneda en mayúsculas.** `Viva cuesta 1200 MXN y sale a las 7:00am.`
  **falla** con `MISATTRIBUTED_FACT`/`UNKNOWN_ENTITY`: `MXN` se interpreta como una entidad desconocida y
  desplaza al dueño de la hora.
- [ ] **El detector de "admisión de falla" es demasiado permisivo.** La lista de palabras (`error`, `no hay`,
  `falló`…) se busca en cualquier parte de la respuesta, de modo que `Sin ningún error` cuenta como admitir
  la falla y puede silenciar `UNRECOVERED_TOOL_ERROR`.
- [ ] **Robustez de la capa de juez:** el contenido evaluado se inserta sin escapar en etiquetas XML
  (un `</answer>` dentro del texto puede romper el delimitador) y `parse_json` toma el **primer** objeto JSON
  de la respuesta, que podría ser uno copiado de la evidencia.

### Huecos de cobertura (el grounding no mira esto)

- Precios **sin `$`** (`500 pesos`, `1200 MXN`): no se reconocen como dato verificable.
- Porcentajes (`30% de descuento`), cantidades en palabras (`mil doscientos`, `las tres de la mañana`) y
  fechas en lenguaje natural.
- Datos reales atribuidos al **campo** equivocado (`$500` de equipaje usado como precio del vuelo): el
  binding es por entidad, no por campo.
- Afirmaciones sin cifras (`incluye desayuno gratis`): solo producen la advertencia léxica `WEAK_SUPPORT`;
  no hacen fallar el reporte. Para rechazarlas hay que pasar un `claim_verifier`, y mientras se corrige el
  punto anterior, protegerlo con tu propio `try/except`.

## Límites de diseño (no son bugs)

- **La evidencia se toma como verdad.** El texto devuelto por las herramientas se trata como fuente confiable;
  una instrucción inyectada en un resultado que la respuesta repita **no** se marca.
- **Razonamiento erróneo con datos respaldados** (`el más barato es Aeromexico $2100`): requiere juez LLM.
- **Heurísticas orientadas a español/inglés y al dominio de ejemplo** (vuelos). Constantes como `GENERIC`,
  `LIGHT`, `OWNER_KINDS` y los `ALIASES` están definidas en el código; adaptarlo a otro dominio implica
  editarlas.
- **`RuleBasedLLM` no es un LLM**: es una política determinista para demos y tests.
- **`AnthropicBackend` no se ha probado contra la API real** (la lógica del juez sí, con un backend
  simulado). Antes de confiar en un juez, mídelo contra etiquetas humanas con `cohen_kappa`.
- **Sin evaluación sobre traces reales** ni benchmarks públicos.

## Uso como librería

```python
from agentcheck import verify, Trace, Agent, RuleBasedLLM, SPECS, ALIASES, RubricJudge, AnthropicBackend

trace = Agent(my_llm).run("pregunta")        # my_llm(messages) -> str; o Trace.from_react_text(...)
report = verify(trace, SPECS, aliases=ALIASES,
                judge=RubricJudge(AnthropicBackend()))   # judge opcional
print(report.passed, report.summary())
```

Si usas un juez o un `claim_verifier`, envuelve `verify()` en un `try/except` hasta que se cierren los dos
primeros bugs de arriba. Para otras herramientas, define tus propios `ToolSpec`/`Param` en lugar de `SPECS`.

