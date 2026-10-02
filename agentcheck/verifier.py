"""Orquestador: trace + herramientas + evidencia => Report."""
from __future__ import annotations

from dataclasses import dataclass, field

from .grounding import build_evidence, check_grounding
from .judge import JudgeError, JudgeResult
from .structure import Issue, check_structure
from .tools import ToolSpec
from .trace import Trace


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    judge: JudgeResult | None = None
    complete: bool = True  # False si un chequeo opcional (juez) no pudo ejecutarse

    @property
    def errors(self): return [i for i in self.issues if i.severity == "error"]
    @property
    def warnings(self): return [i for i in self.issues if i.severity == "warning"]
    @property
    def passed(self) -> bool: return self.complete and not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def summary(self) -> str:
        head = "PASA" if self.passed else ("INCOMPLETO" if not self.complete and not self.errors else "FALLA")
        lines = [f"{head}: {len(self.errors)} errores, {len(self.warnings)} advertencias"]
        lines += [f"  [{i.severity[:4].upper()}] {i.code}: {i.message}" for i in self.issues]
        if self.judge:
            lines.append(f"  juez: {self.judge.scores}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "complete": self.complete,
                "issues": [i.__dict__ for i in self.issues],
                "judge": self.judge.scores if self.judge else None}


def verify(trace: Trace, tools: dict[str, ToolSpec] | None = None, docs=(), aliases=(),
           judge=None, judge_min: int = 3, claim_verifier=None, max_steps: int = 20) -> Report:
    rep = Report(check_structure(trace, tools, max_steps))
    final_idx = next((i for i in range(len(trace.steps) - 1, -1, -1) if trace.steps[i].kind == "final"), None)
    if final_idx is None:
        return rep
    answer = str(trace.steps[final_idx].content)
    ev = build_evidence(trace, docs, aliases)
    rep.issues += check_grounding(answer, ev, claim_verifier, final_idx)
    if judge is not None:
        texts = [d.text for d in ev.docs if d.source == "evidence"]
        try:
            rep.judge = judge.score(trace.question, answer, texts)
            for crit, v in rep.judge.scores.items():
                if v < judge_min:
                    rep.issues.append(Issue("JUDGE_LOW_SCORE", "error",
                                            f"{crit}={v} (< {judge_min}): {rep.judge.reasoning}", final_idx, "judge"))
        except JudgeError as e:
            rep.complete = False
            rep.issues.append(Issue("JUDGE_UNAVAILABLE", "warning", str(e), None, "judge"))
    return rep
