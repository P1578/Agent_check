"""Chequeos estructurales del trace: emparejado llamada/resultado, errores, bucles, cierre."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .text import norm
from .tools import ToolSpec, validate_args
from .trace import Trace

import re

ADMIT = re.compile(r"no pude|no fue posible|no puedo|no encontr|no logr|error|fall[oó]|no hay|unable|couldn.t|failed", re.I)


@dataclass
class Issue:
    code: str
    severity: str  # "error" | "warning"
    message: str
    step: int | None = None
    check: str = "structure"


def check_structure(trace: Trace, tools: dict[str, ToolSpec] | None = None, max_steps: int = 20) -> list[Issue]:
    I: list[Issue] = []
    steps = trace.steps
    if not steps:
        return [Issue("EMPTY_TRACE", "error", "El trace no tiene pasos.")]
    if len(steps) > max_steps:
        I.append(Issue("MAX_STEPS", "error", f"El trace tiene {len(steps)} pasos (máximo {max_steps})."))

    finals = [i for i, s in enumerate(steps) if s.kind == "final"]
    if not finals:
        I.append(Issue("NO_FINAL_ANSWER", "error", "El trace no termina con una respuesta final."))
    else:
        if len(finals) > 1:
            I.append(Issue("MULTIPLE_FINALS", "warning", "Hay más de una respuesta final.", finals[1]))
        if finals[-1] != len(steps) - 1:
            I.append(Issue("STEPS_AFTER_FINAL", "error", "Hay pasos después de la respuesta final.", finals[-1] + 1))

    pending: int | None = None
    pending_msgs: list[tuple[str, str]] = []
    failed: dict[str, int] = {}
    sigs: list[tuple[str, str]] = []
    last_err_sig: tuple[str, str] | None = None

    def dangling(idx):
        I.append(Issue("CALL_WITHOUT_RESULT", "error",
                       f"La llamada a '{steps[idx].tool}' nunca registró un resultado.", idx))
        for code, msg in pending_msgs:
            I.append(Issue(code, "error", msg, idx))

    for i, s in enumerate(steps):
        if s.kind == "tool_call":
            if pending is not None:
                dangling(pending)
            pending, pending_msgs = i, []
            if tools is not None:
                spec = tools.get(s.tool)
                if spec is None:
                    pending_msgs.append(("UNKNOWN_TOOL", f"Herramienta desconocida '{s.tool}'."))
                else:
                    pending_msgs += [("INVALID_ARGS", f"{s.tool}: {e}") for e in validate_args(spec, s.args or {})]
            sig = (s.tool or "", json.dumps(s.args or {}, sort_keys=True))
            sigs.append(sig)
            if sig == last_err_sig:
                I.append(Issue("RETRY_NO_CHANGE", "warning",
                               f"Se repitió '{s.tool}' con los mismos argumentos tras un error.", i))
            if sigs.count(sig) == 3:
                I.append(Issue("LOOP", "warning", f"'{s.tool}' se llamó 3 veces con los mismos argumentos.", i))
        elif s.kind == "tool_result":
            if pending is None:
                I.append(Issue("ORPHAN_RESULT", "error", "Resultado de herramienta sin llamada previa.", i))
                continue
            call = steps[pending]
            sev = "warning" if s.error else "error"  # el runtime rechazó la llamada => recuperable
            for code, msg in pending_msgs:
                I.append(Issue(code, sev, msg, pending))
            if s.error:
                failed[call.tool or ""] = i
                last_err_sig = sigs[-1]
            else:
                failed.pop(call.tool or "", None)
                last_err_sig = None
            pending, pending_msgs = None, []
        elif s.kind == "final" and pending is not None:
            dangling(pending)
            pending, pending_msgs = None, []
    if pending is not None:
        dangling(pending)

    if finals and failed and not ADMIT.search(norm(str(steps[finals[-1]].content))):
        tools_failed = ", ".join(sorted(failed))
        I.append(Issue("UNRECOVERED_TOOL_ERROR", "error",
                       f"La respuesta final se emitió tras un error sin reintento exitoso ({tools_failed}) "
                       "y sin admitir la falla.", finals[-1]))
    return I
