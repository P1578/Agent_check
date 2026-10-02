"""Runtime ReAct real: ejecuta herramientas, valida argumentos y devuelve el error al modelo para que
se autocorrija. Incluye router, herramientas de ejemplo y un 'modelo' determinista para demos/tests."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from .retrieval import BM25
from .text import norm
from .tools import Param, ToolSpec, extract_tool_call, validate_args
from .trace import Step, Trace

LLM = Callable[[list[dict]], str]  # mensajes -> texto del asistente


# ------------------------------------------------------------------ herramientas de ejemplo
ALIASES = [("cun", "cancun"), ("cdmx", "mexico")]
SYNONYMS = [("maleta", "equipaje", "valija"), ("cambio", "cambios", "reprogramar")]
FLIGHTS = [("Viva", 1200, "7:00am", "9:30am"), ("Aeromexico", 2100, "9:30am", "12:00pm"),
           ("Volaris", 1350, "6:00pm", "8:30pm")]
KB = ["Politica de equipaje Viva: 10kg mano gratis, documentada $500 extra",
      "Politica de equipaje Aeromexico: 25kg documentada incluida",
      "Politica de cambios Viva: sin cambios 24h antes, costo $800"]
_bm25 = BM25(KB, SYNONYMS)


def search_flights(origen: str, destino: str, fecha: str) -> list[str]:
    if norm(origen) not in ("cdmx", "mexico") or norm(destino) not in ("cun", "cancun"):
        return []
    return [f"{a}: ${p}, sale {s}, llega {l}" for a, p, s, l in FLIGHTS]


def search_kb(query: str) -> list[str]:
    return [d for d, _ in _bm25.search(query, k=2)]


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., Any]


TOOLS = {
    "search_flights": Tool(ToolSpec("search_flights", {"origen": Param(), "destino": Param(),
                                    "fecha": Param(iso_date=True)}, ["origen", "destino", "fecha"]), search_flights),
    "search_kb": Tool(ToolSpec("search_kb", {"query": Param()}, ["query"]), search_kb),
}
SPECS = {n: t.spec for n, t in TOOLS.items()}


def resolve_date(text: str, today: date) -> str | None:
    n = norm(text)
    for phrase, d in (("pasado manana", 2), ("manana", 1), ("hoy", 0)):
        if re.search(rf"\b{phrase}\b", n):
            return (today + timedelta(days=d)).isoformat()
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return m.group(0) if m else None


# ------------------------------------------------------------------ router
INTENTS = {
    "flight_search": ["vuelo", "vuelos", "volar", "reservar", "boleto", "pasaje", "aerolinea"],
    "knowledge_qa": ["cuanto cuesta", "politica", "equipaje", "maleta", "incluye", "cambios"],
    "lodging": ["hotel", "hospedaje", "airbnb", "alojamiento"],
    "chitchat": ["hola", "quien eres", "chiste", "gracias", "buenos dias"],
}


def route(query: str) -> dict:
    n = norm(query)
    hits = {i: [k for k in kws if re.search(rf"\b{k}\b", n)] for i, kws in INTENTS.items()}
    active = [i for i, h in hits.items() if h and i != "chitchat"]
    if len(active) == 1:
        intent = active[0]
    elif len(active) >= 2:
        intent = "complex_reasoning"  # multi-intención => bucle ReAct con todas las herramientas
    else:
        intent = "chitchat" if hits["chitchat"] else "complex_reasoning"
    return {"intent": intent, "matched": {i: h for i, h in hits.items() if h}}


# ------------------------------------------------------------------ runtime
def _render(c) -> str:
    if isinstance(c, list):
        return "\n".join(map(str, c)) or "Sin resultados"
    return str(c) if c not in (None, "") else "Sin resultados"


class Agent:
    def __init__(self, llm: LLM, tools: dict[str, Tool] | None = None, max_steps: int = 8):
        self.llm, self.tools, self.max_steps = llm, tools or TOOLS, max_steps

    def run(self, question: str) -> Trace:
        msgs: list[dict] = [{"role": "user", "content": question}]
        trace = Trace(question)
        for _ in range(self.max_steps):
            out = self.llm(msgs)
            msgs.append({"role": "assistant", "content": out})
            m = re.search(r"Final Answer\s*:\s*(.*)", out, re.S | re.I)
            if m:
                trace.steps.append(Step("final", m[1].strip()))
                return trace
            call, err = extract_tool_call(out)
            if err:
                trace.steps += [Step("tool_call", tool="<invalid>", args={}),
                                Step("tool_result", f"Error: tool call inválido ({err})", error=True)]
            else:
                name, args = call["tool"], call["parameters"]
                trace.steps.append(Step("tool_call", tool=name, args=args if isinstance(args, dict) else {}))
                trace.steps.append(self._execute(name, args))
            last = trace.steps[-1]
            msgs.append({"role": "user", "content": "Observation: " + _render(last.content)})
        return trace  # sin respuesta final: el verificador lo marcará

    def _execute(self, name: str, args) -> Step:
        tool = self.tools.get(name)
        if tool is None:
            return Step("tool_result", f"Error: herramienta desconocida '{name}'", error=True)
        errs = validate_args(tool.spec, args)
        if errs:
            return Step("tool_result", "Error: " + "; ".join(errs), error=True)
        try:
            return Step("tool_result", tool.fn(**args))
        except Exception as e:  # noqa: BLE001 - el runtime nunca debe caerse por una tool
            return Step("tool_result", f"Error: {type(e).__name__}: {e}", error=True)


class RuleBasedLLM:
    """Política determinista que hace de 'modelo' en demos y tests. NO es un LLM real.
    Intenta primero con la fecha cruda ('mañana'); si la herramienta la rechaza, corrige usando la
    fecha real. La respuesta final se compone a partir de las observaciones (no está escrita a mano)."""

    def __init__(self, today: date):
        self.today = today

    def __call__(self, messages: list[dict]) -> str:
        q = messages[0]["content"]
        obs = [m["content"][len("Observation: "):] for m in messages[1:] if m["role"] == "user"]
        call = lambda t, p: json.dumps({"tool": t, "parameters": p}, ensure_ascii=False)
        if not obs:
            raw = "mañana" if "manana" in norm(q) else (resolve_date(q, self.today) or "mañana")
            return f"Thought: busco vuelos.\n{call('search_flights', {'origen': 'CDMX', 'destino': 'CUN', 'fecha': raw})}"
        if obs[-1].startswith("Error") and "YYYY-MM-DD" in obs[-1]:
            iso = resolve_date(q, self.today) or (self.today + timedelta(days=1)).isoformat()
            return f"Thought: la fecha debe ser ISO.\n{call('search_flights', {'origen': 'CDMX', 'destino': 'CUN', 'fecha': iso})}"
        ok = [o for o in obs if not o.startswith("Error")]
        flights = [l for o in ok for l in o.splitlines() if "llega" in l]
        kb = [l for o in ok for l in o.splitlines() if norm(l).startswith("politica")]
        wants_bag = bool(re.search(r"equipaje|maleta", norm(q)))
        if flights and wants_bag and not kb and not any("politica" in norm(o) for o in ok):
            return call("search_kb", {"query": "equipaje maleta Viva"})
        if not flights:
            return "Final Answer: No encontré vuelos para esa búsqueda."
        rows = [re.match(r"([\w ]+): \$(\d+), sale (\S+),", l) for l in flights]
        best = min((r for r in rows if r), key=lambda r: int(r[2]))
        ans = f"El vuelo más barato es {best[1]} por ${best[2]}, sale a las {best[3]}."
        if kb:
            ans += f" Sobre equipaje: {kb[0]}."
        return "Final Answer: " + ans
