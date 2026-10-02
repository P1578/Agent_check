''' Normalizacion de texto y extracción de hechos verificables (precios, horas, fechas, kg, etc) '''
import...
'''palabras de relleno (las que se me ocurrieron)'''

STOP = frozenset("""a al algo ante aun con como cual cuales cuando
                cuanto cuanta de del donde el ella ellos en es esa ese eso
                esta este esto ha han hay la las le les lo los me mi mas muy
                no o para por pero que si sin sobre son su sus te tiene tu un una 
                uno unos y ya the of and to in is it for on with you your are be as at by 
                an or that this from can will was were has have""". split())


'''palabras de no tan relleno'''
LIGHT = frozenset("""cuesta cuestan costo precio precios incluye incluyen 
                    encontre encontramos opcion opciones recomiendo puedes 
                    quieres disponible disponibles aqui tienes""".split())

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize ("NFKD", s) if not unicodedata.combining(c))

def norm (s: str) -> str:
    """Minúsculas sin acentos; conserva la longitud (los indices sirven en el texto original)."""
    return strip_accents(s).lower()

''' sufijos '''

_SUFFIXES = ("aciones", "acion", "ciones", "cion", "mente", "iendo", "ando", "adas", "ados",
             "ada", "ado", "idas", "idos", "ida", "ido", "ar", "er", "ir")

def stem(w: str) -> str:
    """ Stemer tosco para español e ingles : plural, sufijos, vocal final
    trunca a 6 letras, es consistente entre consulta y documento """

    if len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w)>3 and w.endswith("s"):
        w = w[:-1]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >=4 :
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w[-1] in "aeo":
        w = w[:-1] 
    return w[:6]


def tokens(s:str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(s))       

class Fact(NamedTuple):
    kind: str 
    value: float | int| str
    start: int
    end: int

_FACT_RE = re.compile(
    r"(?P<money>\$\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?P<date>\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?P<t12>\b\d{1,2}(?::\d{2})?\s?[ap]\.?m\.?(?![a-z]))"
