import json
import sys
from datetime import date

from . import ALIASES, SPECS, Agent, RuleBasedLLM, Trace, verify
from .evalset import evaluate, format_report


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd == "eval":
        res = evaluate()
        print(format_report(res))
        return 1 if res["mismatches"] else 0
    if cmd == "demo":
        q = " ".join(argv[2:]) or "Busca vuelo barato CDMX-Cancún para mañana y dime equipaje"
        trace = Agent(RuleBasedLLM(date.today())).run(q)
        for s in trace.steps:
            print(f"  {s.kind:<12} {s.tool or ''} {s.args or ''} {s.content if s.content is not None else ''}")
        print("\n>> Verificación del trace real:\n" + verify(trace, SPECS, aliases=ALIASES).summary())
        bad = Trace.from_dict(trace.to_dict())
        bad.steps[-1].content = "Aeromexico cuesta $1200 y sale a las 3:00am."
        print("\n>> Mismo trace con una respuesta alucinada:\n" + verify(bad, SPECS, aliases=ALIASES).summary())
        return 0
    if cmd == "verify" and len(argv) > 2:
        rep = verify(Trace.from_dict(json.load(open(argv[2]))), SPECS, aliases=ALIASES)
        print(rep.summary())
        return 0 if rep.passed else 1
    print("uso: python -m agentcheck [eval | demo [pregunta] | verify trace.json]")
    return 2


sys.exit(main(sys.argv))
