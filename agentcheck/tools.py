"""Especificación y validación de tool calls (schema mínimo, sin dependencias)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


@dataclass
class Param:
    type: str = "string"
    enum: list | None = None
    pattern: str | None = None
    iso_date: bool = False


@dataclass
class ToolSpec:
    name: str
    params: dict[str, Param]
    required: list[str] = field(default_factory=list)


def validate_args(spec: ToolSpec, args) -> list[str]:
    if not isinstance(args, dict):
        return ["'parameters' debe ser un objeto JSON"]
    errs = [f"falta el parámetro obligatorio '{r}'" for r in spec.required if r not in args]
    for k, v in args.items():
        p = spec.params.get(k)
        if p is None:
            errs.append(f"parámetro desconocido '{k}'")
            continue
        bad_bool = isinstance(v, bool) and p.type in ("integer", "number")
        if bad_bool or not isinstance(v, _TYPES[p.type]):
            errs.append(f"'{k}' debe ser {p.type} (recibí {type(v).__name__})")
            continue
        if p.enum is not None and v not in p.enum:
            errs.append(f"'{k}' debe ser uno de {p.enum} (recibí {v!r})")
        if p.pattern and not re.fullmatch(p.pattern, v):
            errs.append(f"'{k}' no cumple el patrón {p.pattern!r} (recibí {v!r})")
        if p.iso_date:
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                    raise ValueError
                date.fromisoformat(v)
            except ValueError:
                errs.append(f"'{k}' debe tener formato YYYY-MM-DD (recibí {v!r})")
    return errs


def extract_tool_call(text: str) -> tuple[dict | None, str | None]:
    """Busca el primer objeto JSON *balanceado* con 'tool' (o 'name'). Devuelve (call, error)."""
    dec, last_err = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
            name = obj.get("tool", obj.get("name"))
            params = obj.get("parameters", obj.get("arguments"))
            if not isinstance(name, str):
                return None, "'tool' debe ser un string"
            if params is None:
                return None, "falta 'parameters'"
            return {"tool": name, "parameters": params}, None
    return None, (f"JSON inválido: {last_err}" if last_err else "no se encontró un tool call JSON")
