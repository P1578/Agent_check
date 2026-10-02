"""Conjunto etiquetado para medir al propio verificador (precisión/recall) y ver qué NO puede detectar.

Los casos los escribió una persona: sirven como regresión y para documentar límites, NO como prueba de
generalización. Los casos tier='semantic' están etiquetados 'fail' pero son indetectables con chequeos
estáticos (los datos están respaldados; el error es de razonamiento): requieren juez LLM.
"""
from __future__ import annotations

from .agent import ALIASES, SPECS
from .trace import Step, Trace
from .verifier import verify

Q = "Busca vuelo barato CDMX-Cancún para mañana y dime equipaje"
CALL = ("c", "search_flights", {"origen": "CDMX", "destino": "CUN", "fecha": "2026-09-29"})
BADCALL = ("c", "search_flights", {"origen": "CDMX", "destino": "CUN", "fecha": "mañana"})
R1 = ("r", ["Viva: $1200, sale 7:00am, llega 9:30am", "Aeromexico: $2100, sale 9:30am, llega 12:00pm"])
KBC = ("c", "search_kb", {"query": "equipaje Viva"})
KBR = ("r", "Politica de equipaje Viva: 10kg mano gratis, documentada $500 extra")
ERR = ("e", "Error: timeout")
F = lambda t: ("f", t)  # noqa: E731
T = ("t", "Voy a buscar.")

CASES = [
    # ---- deben PASAR
    ("good_basic", "static", "pass", [T, CALL, R1, F("El vuelo más barato es Viva por $1200, sale a las 7:00am.")]),
    ("good_two_options", "static", "pass", [CALL, R1, F("Encontré Viva $1200 (7am) y Aeromexico $2100 (9:30am). ¿Quieres que reserve el de Viva?")]),
    ("good_policy", "static", "pass", [CALL, R1, KBC, KBR, F("Viva por $1200. Viva incluye 10kg de mano gratis; documentar cuesta $500 extra.")]),
    ("good_recovery", "static", "pass", [BADCALL, ("e", "Error: 'fecha' debe tener formato YYYY-MM-DD (recibí 'mañana')"), CALL, R1, F("Viva cuesta $1200 y sale a las 7:00am.")]),
    ("honest_failure", "static", "pass", [CALL, ERR, F("No pude consultar los vuelos por un error del servicio.")]),
    ("smalltalk", "static", "pass", [T, F("Hola, ¿en qué te puedo ayudar?")]),
    ("good_24h_format", "static", "pass", [CALL, R1, F("Aeromexico cuesta $2,100 y sale a las 09:30.")]),
    # ---- deben FALLAR (estáticos)
    ("wrong_price", "static", "fail", [CALL, R1, F("El vuelo Viva cuesta $500 y sale a las 7:00am.")]),
    ("misattributed_price", "static", "fail", [CALL, R1, F("Aeromexico cuesta $1200 y sale a las 7:00am.")]),
    ("invented_airline", "static", "fail", [CALL, R1, F("Volaris cuesta $800, sale a las 3:00am, con desayuno gratis.")]),
    ("fabricated_time", "static", "fail", [CALL, R1, F("Viva cuesta $1200 y sale a las 3:00am.")]),
    ("substring_price", "static", "fail", [CALL, R1, F("Viva cuesta $120.")]),
    ("wrong_date", "static", "fail", [CALL, R1, F("Encontré Viva por $1200 para el 2026-10-15.")]),
    ("no_evidence", "static", "fail", [T, F("El vuelo Viva cuesta $1200.")]),
    ("no_final", "static", "fail", [T, CALL, R1]),
    ("call_without_result", "static", "fail", [T, CALL, F("Viva cuesta $1200.")]),
    ("orphan_result", "static", "fail", [R1, F("El vuelo más barato es Viva por $1200.")]),
    ("steps_after_final", "static", "fail", [CALL, R1, F("Viva cuesta $1200."), T]),
    ("ignores_error", "static", "fail", [CALL, ERR, F("El vuelo más barato es Viva por $1200.")]),
    ("unknown_tool_accepted", "static", "fail", [("c", "book_flight", {"id": "Viva123"}), ("r", "Reserva confirmada"), F("Reservé el vuelo.")]),
    ("invalid_args_accepted", "static", "fail", [BADCALL, R1, F("El vuelo más barato es Viva por $1200.")]),
    ("retry_loop", "static", "fail", [CALL, ERR, CALL, ERR, CALL, ERR, F("Listo, aquí tienes tu vuelo.")]),
    # ---- semánticos: datos respaldados, razonamiento erróneo (requieren juez)
    ("wrong_comparison", "semantic", "fail", [CALL, R1, F("El vuelo más barato es Aeromexico por $2100, sale a las 9:30am.")]),
    ("ignores_question", "semantic", "fail", [CALL, R1, F("Aeromexico sale a las 9:30am.")]),
    ("fee_as_price", "semantic", "fail", [CALL, R1, KBC, KBR, F("El vuelo Viva cuesta $500.")]),  # dato real, campo equivocado
    ("invented_perk", "semantic", "fail", [CALL, R1, F("El vuelo Viva cuesta $1200 e incluye desayuno gratis en el trayecto.")]),
]


def build(steps: list) -> Trace:
    out = []
    for s in steps:
        k = s[0]
        if k == "t": out.append(Step("thought", s[1]))
        elif k == "c": out.append(Step("tool_call", tool=s[1], args=s[2]))
        elif k == "r": out.append(Step("tool_result", s[1]))
        elif k == "e": out.append(Step("tool_result", s[1], error=True))
        else: out.append(Step("final", s[1]))
    return Trace(Q, out)


def evaluate(judge=None) -> dict:
    rows = []
    for cid, tier, label, steps in CASES:
        rep = verify(build(steps), SPECS, aliases=ALIASES, judge=judge)
        rows.append({"id": cid, "tier": tier, "label": label, "pred": "pass" if rep.passed else "fail",
                     "codes": sorted(rep.codes() - {"WEAK_SUPPORT"} if rep.passed else rep.codes())})
    st = [r for r in rows if r["tier"] == "static"]
    tp = sum(r["label"] == "fail" and r["pred"] == "fail" for r in st)
    fn = sum(r["label"] == "fail" and r["pred"] == "pass" for r in st)
    fp = sum(r["label"] == "pass" and r["pred"] == "fail" for r in st)
    tn = sum(r["label"] == "pass" and r["pred"] == "pass" for r in st)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    sem = [r for r in rows if r["tier"] == "semantic"]
    return {"rows": rows, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "semantic_detected": sum(r["pred"] == "fail" for r in sem), "semantic_total": len(sem),
            "mismatches": [r for r in st if r["label"] != r["pred"]]}


def format_report(res: dict) -> str:
    L = [f"{'caso':<24}{'tier':<10}{'etiqueta':<9}{'pred':<6}códigos"]
    for r in res["rows"]:
        mark = "" if r["label"] == r["pred"] else "  <-- FALLO DEL VERIFICADOR" if r["tier"] == "static" else "  (no detectable estáticamente)"
        L.append(f"{r['id']:<24}{r['tier']:<10}{r['label']:<9}{r['pred']:<6}{','.join(r['codes'])}{mark}")
    L.append(f"\nEstático  TP={res['tp']} FN={res['fn']} FP={res['fp']} TN={res['tn']}  "
             f"precision={res['precision']:.2f} recall={res['recall']:.2f} F1={res['f1']:.2f}")
    L.append(f"Semántico: {res['semantic_detected']}/{res['semantic_total']} detectados sin juez LLM")
    return "\n".join(L)
