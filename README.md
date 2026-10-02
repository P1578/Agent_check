# agentcheck

Verificador de traces de agentes de IA. Sin dependencias (solo stdlib).

```bash
python -m agentcheck eval            # mide al verificador: precisión/recall + qué NO detecta
python -m agentcheck demo            # corre un agente real (ReAct), lo verifica y luego lo "alucina"
python -m agentcheck verify t.json   # verifica un trace guardado ({"question","steps":[...]})
python -m pytest                     # 29 tests
```

## Qué verifica (y cómo)

| Capa | Códigos | Cómo |
|---|---|---|
| Estructura | `NO_FINAL_ANSWER`, `CALL_WITHOUT_RESULT`, `ORPHAN_RESULT`, `STEPS_AFTER_FINAL`, `UNKNOWN_TOOL`, `INVALID_ARGS`, `UNRECOVERED_TOOL_ERROR`, `RETRY_NO_CHANGE`, `LOOP` | Recorrido **secuencial** del trace; args validados contra `ToolSpec`. Un error de herramienta *recuperado* es advertencia; uno ignorado es error. |
| Grounding | `FABRICATED_FACT`, `MISATTRIBUTED_FACT`, `UNKNOWN_ENTITY`, `WEAK_SUPPORT` | Los datos (precios, horas, fechas, kg) de la respuesta deben salir de lo que devolvieron las herramientas, y pertenecer a la entidad correcta. `$1,200` = `$1200`; `7am` = `07:00`; `$120` ≠ `$1200`. |
| Juez (opcional) | `JUDGE_LOW_SCORE`, `JUDGE_UNAVAILABLE` | Rúbrica con salida JSON validada; pairwise en ambos órdenes (detecta sesgo de posición); si el juez falla, el reporte queda **incompleto**, nunca "pasa". |

## Límites conocidos (medidos, no supuestos)

`python -m agentcheck eval` reporta 15/15 fallos estáticos detectados y 0 falsos positivos **sobre 22 casos escritos por una persona**: sirve como regresión, no como prueba de generalización. Además hay casos `semantic` que el verificador **no** detecta sin juez LLM:

- Razonamiento erróneo con datos respaldados ("el más barato es Aeromexico $2100").
- Dato real en el campo equivocado (`$500` es la tarifa de equipaje de Viva, no el precio del vuelo): el binding es por entidad, no por campo.
- Afirmaciones sin cifras ("incluye desayuno gratis"): solo advertencia léxica; para rechazarlas pasa un `claim_verifier` (`LLMClaimVerifier`).

Otras limitaciones: heurísticas pensadas para español/inglés; `RuleBasedLLM` **no es un LLM** (es una política determinista para demos/tests); `AnthropicBackend` no se probó contra la API real (la lógica del juez sí, con un backend simulado). Antes de confiar en un juez, mídelo contra etiquetas humanas con `cohen_kappa`.

## Uso como librería

```python
from agentcheck import verify, Trace, Agent, RuleBasedLLM, SPECS, ALIASES, RubricJudge, AnthropicBackend

trace = Agent(my_llm).run("pregunta")        # my_llm(messages) -> str; o Trace.from_react_text(...)
report = verify(trace, SPECS, aliases=ALIASES,
                judge=RubricJudge(AnthropicBackend()))   # judge opcional
print(report.passed, report.summary())
```

## Mapa desde el repo anterior

`validate_trace.py`→`structure.py` · `hallucination_detector.py`→`grounding.py` · `tool_call_parser.py`→`tools.py` ·
`mini_rag.py`→`retrieval.py` (BM25) · `llm_judge.py`→`judge.py` · `react_agent.py`+`self_correction.py`→`agent.py`
(el error de herramienta ahora es real y se devuelve al modelo) · `router.py`→`agent.route` · `final_agent.py`→`__main__ demo`.
`planner.py` se eliminó: eran plantillas fijas sin lógica; el bucle ReAct lo reemplaza.
