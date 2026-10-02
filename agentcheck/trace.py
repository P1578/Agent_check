"""Modelo de datos de un trace de agente."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

KINDS = ("thought", "tool_call", "tool_result", "final")


@dataclass
class Step:
    kind: str
    content: Any = None
    tool: str | None = None
    args: dict | None = None
    error: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind inválido: {self.kind!r}; debe ser uno de {KINDS}")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, False)}


@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)

    @property
    def final(self) -> str | None:
        fin = [s for s in self.steps if s.kind == "final"]
        return str(fin[-1].content) if fin else None

    def to_dict(self) -> dict:
        return {"question": self.question, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        return cls(d.get("question", ""), [Step(**s) for s in d.get("steps", [])])

    @classmethod
    def from_react_text(cls, question: str, text: str) -> "Trace":
        """Parsea el formato clásico ReAct: Thought / Action / Observation / Final Answer."""
        blocks: list[list[str]] = []
        for line in text.splitlines():
            m = re.match(r"\s*(Thought|Action|Observation|Final Answer)\s*:\s*(.*)", line, re.I)
            if m:
                blocks.append([m[1].lower(), m[2]])
            elif blocks:
                blocks[-1][1] += "\n" + line
        steps: list[Step] = []
        for label, body in blocks:
            body = body.strip()
            if label == "thought":
                steps.append(Step("thought", body))
            elif label == "final answer":
                steps.append(Step("final", body))
            elif label == "observation":
                steps.append(Step("tool_result", body, error=body.lower().startswith("error")))
            else:  # action: nombre [json]
                m = re.match(r"(\w+)\s*\(?\s*(\{.*\})?\s*\)?\s*$", body, re.S)
                name, raw = (m[1], m[2]) if m else (body, None)
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {}
                steps.append(Step("tool_call", tool=name, args=args))
        return cls(question, steps)
