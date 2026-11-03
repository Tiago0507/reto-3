"""Base de conocimiento: reglamento de pregrado de Icesi y guía de cancelación de semestre.

El corpus se genera desde los PDF oficiales con `asistente-icesi-preparar` y se
guarda en `datos/corpus.json`: un fragmento por artículo (`Art. N`) y por sección
de la guía (`Guía N`). La búsqueda es BM25 sobre términos normalizados; en
producción podría reemplazarse por un índice vectorial sin tocar a los agentes.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from functools import cache
from typing import TypedDict

from .configuracion import RAIZ_PROYECTO

RUTA_CORPUS = RAIZ_PROYECTO / "datos" / "corpus.json"


class Fragmento(TypedDict):
    id: str  # "Art. 53" o "Guía 2"
    fuente: str
    capitulo: str
    texto: str


@cache
def corpus() -> tuple[Fragmento, ...]:
    if not RUTA_CORPUS.exists():
        raise FileNotFoundError(f"No existe {RUTA_CORPUS}. Ejecuta: uv run asistente-icesi-preparar")
    return tuple(json.loads(RUTA_CORPUS.read_text(encoding="utf-8")))


@cache
def por_id() -> dict[str, Fragmento]:
    return {f["id"]: f for f in corpus()}


def normalizar_id(texto: str) -> str | None:
    """'artículo 53', 'Art 53', '53' → 'Art. 53'; 'guía 2' → 'Guía 2'. None si no se reconoce."""
    t = normalizar(texto).strip()
    if m := re.fullmatch(r"(?:art(?:iculo)?\.?\s*)?(\d+)", t):
        return f"Art. {int(m.group(1))}"
    if m := re.fullmatch(r"guia\.?\s*(\d+)", t):
        return f"Guía {int(m.group(1))}"
    return None


# --- Búsqueda ---------------------------------------------------------------

_STOP = frozenset(
    "de la el los las un una unos unas y o u en a por para con que se del al es su sus como mas cual cuales "
    "hay debe deberia puede puedo si sin lo le les ante sobre entre esta este estos estas son ser sera "
    "tiene tengo mi mis me yo cuando donde cuanto cuantos cuantas pasa hacer hago quiero necesito".split()
)


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


_SUFIJOS = ("amientos", "imientos", "amiento", "imiento", "aciones", "iciones", "acion", "icion", "arse", "erse",
            "irse", "ados", "adas", "idos", "idas", "ado", "ada", "ido", "ida", "ar", "er", "ir", "es", "os", "as",
            "s", "o", "a", "e")


def _raiz(t: str) -> str:
    """Raíz simple para el español: 'retirado', 'retiro' y 'retirar' → 'retir'."""
    for sufijo in _SUFIJOS:
        if t.endswith(sufijo) and len(t) - len(sufijo) >= 4:
            return t[: -len(sufijo)]
    return t


def terminos(texto: str) -> list[str]:
    return [_raiz(t) for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if t not in _STOP and len(t) > 1]


@cache
def _indice() -> tuple[list[Counter], dict[str, float], float]:
    docs = [Counter(terminos(f"{f['fuente']} {f['capitulo']} {f['texto']}")) for f in corpus()]
    n = len(docs)
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    promedio = sum(sum(d.values()) for d in docs) / n
    return docs, idf, promedio


def buscar(consulta: str, k: int = 4, k1: float = 1.4, b: float = 0.6) -> list[Fragmento]:
    """BM25: premia los términos raros y no favorece de más a los artículos largos."""
    q = set(terminos(consulta))
    docs, idf, promedio = _indice()
    puntajes = []
    for i, d in enumerate(docs):
        largo = sum(d.values())
        s = sum(idf[t] * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * largo / promedio)) for t in q if t in d)
        if s > 0:
            puntajes.append((s, i))
    puntajes.sort(key=lambda p: (-p[0], p[1]))
    return [corpus()[i] for _, i in puntajes[:k]]
