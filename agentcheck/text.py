"""Normalización de texto y extracción de hechos verificables (precios, horas, fechas, kg)."""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

STOP = frozenset("""a al algo ante aun con como cual cuales cuando cuanto cuanta de del donde el ella ellos en es esa ese eso
esta este esto ha han hay la las le les lo los me mi mas muy no o para por pero que se si sin sobre son su sus te tiene tu un una
uno unos y ya the of and to in is it for on with you your are be as at by an or that this from can will was were has have""".split())

# Palabras "de relleno" que casi nunca son un dato verificable por sí mismas.
LIGHT = frozenset("""cuesta cuestan costo precio precios incluye incluyen encontre encontramos opcion opciones recomiendo
puedes quieres disponible disponibles aqui tienes""".split())


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s: str) -> str:
    """Minúsculas sin acentos; conserva la longitud (los índices sirven en el texto original)."""
    return strip_accents(s).lower()


_SUFFIXES = ("aciones", "acion", "ciones", "cion", "mente", "iendo", "ando", "adas", "ados", "ada", "ado",
             "idas", "idos", "ida", "ido", "ar", "er", "ir")


def stem(w: str) -> str:
    """Stemmer tosco para español/inglés: plural, sufijos verbales, vocal final; trunca a 6 letras.
    Es consistente entre consulta y documento (lo que importa aquí), no lingüísticamente correcto."""
    if len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w[-1] in "aeo":
        w = w[:-1]
    return w[:6]


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(s))


class Fact(NamedTuple):
    kind: str  # money | time | date | kg
    value: float | int | str
    start: int
    end: int


_FACT_RE = re.compile(
    r"(?P<money>\$\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?P<date>\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?P<t12>\b\d{1,2}(?::\d{2})?\s?[ap]\.?m\.?(?![a-z]))"
    r"|(?P<t24>\b(?:[01]?\d|2[0-3]):[0-5]\d\b)"
    r"|(?P<kg>\b\d+(?:\.\d+)?\s?kg\b)",
    re.I,
)


def extract_facts(text: str) -> list[Fact]:
    t = norm(text)
    out: list[Fact] = []
    for m in _FACT_RE.finditer(t):
        k, raw = m.lastgroup, m.group(0)
        if k == "money":
            out.append(Fact("money", float(re.sub(r"[^\d.]", "", raw)), m.start(), m.end()))
        elif k == "date":
            out.append(Fact("date", raw, m.start(), m.end()))
        elif k == "t12":
            mm = re.match(r"(\d{1,2})(?::(\d{2}))?\s?([ap])", raw)
            mins = (int(mm[1]) % 12 + (12 if mm[3] == "p" else 0)) * 60 + int(mm[2] or 0)
            out.append(Fact("time", mins, m.start(), m.end()))
        elif k == "t24":
            h, mi = raw.split(":")
            out.append(Fact("time", int(h) * 60 + int(mi), m.start(), m.end()))
        else:
            out.append(Fact("kg", float(re.match(r"[\d.]+", raw)[0]), m.start(), m.end()))
    return out


def mask_facts(text: str) -> str:
    t = list(norm(text))
    for f in extract_facts(text):
        for i in range(f.start, f.end):
            t[i] = " "
    return "".join(t)


def content_words(text: str, ignore: frozenset[str] = LIGHT) -> list[str]:
    return [
        stem(t)
        for t in re.findall(r"[a-z0-9]+", mask_facts(text))
        if t not in STOP and t not in ignore and not t.isdigit() and len(t) > 2
    ]


def doc_numbers(text: str) -> set[float]:
    """Números 'sueltos' de un documento, sin contar horas ni fechas (evita que '7:00' respalde '$7')."""
    t = list(norm(text))
    for f in extract_facts(text):
        if f.kind in ("time", "date"):
            for i in range(f.start, f.end):
                t[i] = " "
    return {float(x.rstrip(",").replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", "".join(t))}
