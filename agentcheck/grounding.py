"""Grounding: ¿los datos de la respuesta final salen de la evidencia del trace?

Chequeos (heurísticos, determinísticos):
  FABRICATED_FACT     precio/hora/fecha/kg que no aparece en ninguna evidencia
  MISATTRIBUTED_FACT  el dato existe, pero pertenece a otra entidad (ej. precio de Viva atribuido a Aeromexico)
  UNKNOWN_ENTITY      nombre propio que no aparece en la evidencia (warning)
  WEAK_SUPPORT        oración con poco vocabulario respaldado por la evidencia (warning, o error si un
                      `claim_verifier` -LLM o NLI- la rechaza)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .structure import Issue
from .text import STOP, content_words, doc_numbers, extract_facts, norm, stem, tokens
from .trace import Trace

GENERIC = frozenset("vuelo vuelos politica politicas precio precios salida llegada aerolinea resultado resultados "
                    "encontrados error cambios equipaje paso final".split())
CAP_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]{2,}\b")
OWNER_KINDS = ("money", "time", "kg")


@dataclass
class Doc:
    text: str
    source: str = "evidence"  # evidence | user | args | error
    ents: set[str] = field(default_factory=set)
    facts: list = field(default_factory=list)
    nums: set = field(default_factory=set)


def flatten(c) -> list[str]:
    if c is None:
        return []
    if isinstance(c, str):
        return [l for l in c.splitlines() if l.strip()]
    if isinstance(c, (int, float, bool)):
        return [str(c)]
    if isinstance(c, dict):
        parts = [f"{k}: {' '.join(flatten(v)) if isinstance(v, (dict, list)) else v}" for k, v in c.items()]
        return [", ".join(parts)]
    return [t for x in c for t in flatten(x)]


class Evidence:
    def __init__(self, docs: list[Doc], aliases: Iterable[Iterable[str]] = ()):
        self.docs = docs
        self.canon: dict[str, str] = {}
        for g in aliases:
            g = [norm(x) for x in g]
            for w in g:
                self.canon[w] = g[0]
        lower_seen = {norm(w) for d in docs for w in re.findall(r"\b[a-záéíóúñ]{3,}\b", d.text)}
        self.known_words = {t for d in docs for t in tokens(d.text)} | set(self.canon)
        self.words = {w for d in docs for w in content_words(d.text)}
        self.surface: dict[str, str] = {}
        for d in docs:
            d.facts, d.nums = extract_facts(d.text), doc_numbers(d.text)
            if d.source in ("error", "args"):
                continue
            for m in CAP_RE.finditer(d.text):
                w = norm(m.group(0))
                if w in STOP or w in GENERIC or w in lower_seen or (m.group(0).isupper() and len(w) > 4):
                    continue
                key = self.canon.get(w, w)
                d.ents.add(key)
                if d.source == "evidence":  # solo la evidencia puede 'poseer' datos; lo de la pregunta es contexto
                    self.surface[w] = key
        owners = set(self.surface.values())
        for w, key in self.canon.items():  # sinónimos de entidades-dueñas conocidas
            if key in owners:
                self.surface[w] = key

    def fact_in(self, f, d: Doc) -> bool:
        if f.kind in ("money", "kg"):
            return f.value in d.nums
        return any(x.kind == f.kind and x.value == f.value for x in d.facts)


def build_evidence(trace: Trace, extra_docs: Iterable[str] = (), aliases=()) -> Evidence:
    docs = [Doc(trace.question, "user")]
    for s in trace.steps:
        if s.kind == "tool_result":
            docs += [Doc(t, "error" if s.error else "evidence") for t in flatten(s.content)]
        elif s.kind == "tool_call" and s.args:
            docs.append(Doc(json.dumps(s.args, ensure_ascii=False), "args"))
    docs += [Doc(t, "evidence") for t in extra_docs]
    return Evidence(docs, aliases)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def check_grounding(answer: str, ev: Evidence,
                    claim_verifier: Callable[[str, list[str]], bool] | None = None,
                    step: int | None = None) -> list[Issue]:
    issues: list[Issue] = []
    seen_unknown: set[str] = set()
    prev_owner: tuple[str, bool] | None = None
    texts = [d.text for d in ev.docs if d.source in ("evidence", "user")]

    for s in split_sentences(answer):
        if s.endswith("?") or s.startswith("¿"):
            continue  # preguntas/ofertas no afirman hechos
        n = norm(s)
        mentions: list[tuple[int, str, bool]] = []
        for w, key in ev.surface.items():
            mentions += [(m.start(), key, True) for m in re.finditer(rf"\b{re.escape(w)}\b", n)]
        for m in CAP_RE.finditer(s):
            w = norm(m.group(0))
            initial = s[:m.start()].strip(" ¡¿\"'(*-") == ""
            if (initial or w in ev.known_words or w in STOP or w in GENERIC
                    or (m.group(0).isupper() and len(w) > 4)):
                continue
            mentions.append((m.start(), w, False))
            if w not in seen_unknown:
                seen_unknown.add(w)
                issues.append(Issue("UNKNOWN_ENTITY", "warning",
                                    f"'{m.group(0)}' no aparece en la evidencia.", step, "grounding"))
        mentions.sort()

        for f in extract_facts(s):
            label = s[f.start:f.end]
            pool = [d for d in ev.docs if d.source in ("evidence", "user") or (f.kind == "date" and d.source == "args")]
            holders = [d for d in pool if ev.fact_in(f, d)]
            if not holders:
                issues.append(Issue("FABRICATED_FACT", "error",
                                    f"'{label}' no aparece en ninguna evidencia del trace.", step, "grounding"))
                continue
            if f.kind not in OWNER_KINDS or any(d.source == "user" for d in holders):
                continue
            before = [m for m in mentions if m[0] < f.start]
            after = [m for m in mentions if m[0] >= f.start]
            owner = (before[-1] if before else after[0])[1:] if mentions else prev_owner  # preferir la precedente
            if owner and not (owner[1] and any(owner[0] in d.ents for d in holders)):
                who = owner[0]
                issues.append(Issue("MISATTRIBUTED_FACT", "error",
                                    f"'{label}' existe en la evidencia pero no pertenece a '{who}'.", step, "grounding"))
        if mentions:
            prev_owner = mentions[-1][1:]

        cw = content_words(s)
        if len(cw) >= 3 and sum(w in ev.words for w in cw) / len(cw) < 0.5:
            if claim_verifier is None:
                issues.append(Issue("WEAK_SUPPORT", "warning",
                                    f"Poco respaldo léxico en la evidencia: «{s}»", step, "grounding"))
            elif not claim_verifier(s, texts):
                issues.append(Issue("UNSUPPORTED_CLAIM", "error",
                                    f"El verificador de claims rechazó: «{s}»", step, "grounding"))
    return issues
