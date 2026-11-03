"""Convierte los PDF oficiales de Icesi en el corpus del asistente (`datos/corpus.json`).

Fuentes (descargadas de icesi.edu.co, en `datos/fuentes/`):
- Resolución de Rectoría 1.690 del 24 de marzo de 2026: texto completo y vigente del
  Libro de Derechos, Deberes y Normas de los Estudiantes de Pregrado (110 artículos).
- Guía del servicio "Cancelación de semestre pregrado - Estudiantes" (Banner).

Cada artículo queda como un fragmento con id `Art. N`; la guía, como `Guía N`.
Se ejecuta una sola vez (o cuando cambien las fuentes):

    uv run asistente-icesi-preparar
"""

from __future__ import annotations

import json
import re
from collections import Counter

from pypdf import PdfReader

from .configuracion import RAIZ_PROYECTO

FUENTES = RAIZ_PROYECTO / "datos" / "fuentes"
SALIDA = RAIZ_PROYECTO / "datos" / "corpus.json"

REGLAMENTO = "reglamento_pregrado_2026.pdf"
GUIA = "guia_cancelacion_semestre.pdf"
NOMBRE_REGLAMENTO = "Libro de Derechos, Deberes y Normas de los Estudiantes de Pregrado (Res. 1.690/2026)"
NOMBRE_GUIA = "Guía del servicio de cancelación de semestre (Banner)"

_PIE = re.compile(r"^\s*(Resolución de Rectoría No\. 1\.690/2026|Página \d+ de \d+)\s*$", re.M)
_ARTICULO = re.compile(r"^\s*ARTÍCULO\s+(\d+)\s*\.\s*", re.M)
_CAPITULO = re.compile(r"^\s*CAPÍTULO\s+([IVXLl]+)\s*\.", re.M)
_INDICE = re.compile(r"^\s*Capítulo ([IVXL]+): (.+?)\s*$", re.M)


def _limpiar(texto: str) -> str:
    """Une las líneas cortadas por el PDF y conserva los párrafos."""
    texto = _PIE.sub("", texto)
    parrafos = re.split(r"\n\s*\n", texto)
    limpios = (re.sub(r"\s+", " ", p).strip() for p in parrafos)
    return "\n".join(p for p in limpios if p)


def _reparar_palabras(fragmentos: list[dict]) -> None:
    """Une palabras partidas por el PDF ("exa men" -> "examen") si la forma unida existe en el corpus."""
    frecuencia = Counter(w for f in fragmentos for w in re.findall(r"\w+", f["texto"].lower()))

    def conocida(w: str) -> bool:  # una pieza rota aparece una sola vez; una palabra real, varias
        return frecuencia[w.lower()] >= 2

    for f in fragmentos:
        partes = re.split(r"( )", f["texto"])  # conserva los separadores
        k = 0
        while k + 2 < len(partes):
            a, b = partes[k], partes[k + 2]
            if a.isalpha() and b.isalpha() and conocida(a + b) and not (conocida(a) and conocida(b)):
                partes[k : k + 3] = [a + b]
            else:
                k += 2
        f["texto"] = "".join(partes)


# El cuadro del Art. 68 pierde su forma al extraerse; se reescribe como texto explícito.
_CUADRO_68 = (r"Materias cursadas en el período académico Materias perdidas en el período 1 1 2 2 3 o 4 2 5 o más 3",
              "(cuadro: si cursa 1 materia y pierde 1; si cursa 2 materias y pierde 2; si cursa 3 o 4 materias y "
              "pierde 2 o más; si cursa 5 o más materias y pierde 3 o más)")


def _texto_pdf(nombre: str, modo: str) -> list[str]:
    return [p.extract_text(extraction_mode=modo) for p in PdfReader(FUENTES / nombre).pages]


def fragmentos_reglamento() -> list[dict]:
    texto = "\n".join(_texto_pdf(REGLAMENTO, "layout"))  # modo layout: no parte palabras
    titulos = {num: titulo for num, titulo in _INDICE.findall(texto)}
    inicio = _CAPITULO.search(texto).start()
    cuerpo = texto[inicio: texto.index("HASTA AQUÍ EL REGLAMENTO")]

    capitulos = [(m.start(), m.group(1).replace("l", "I")) for m in _CAPITULO.finditer(cuerpo)]
    articulos = list(_ARTICULO.finditer(cuerpo))
    salida = []
    for i, m in enumerate(articulos):
        fin = articulos[i + 1].start() if i + 1 < len(articulos) else len(cuerpo)
        crudo = _CAPITULO.split(cuerpo[m.end(): fin])[0]  # quita el encabezado del capítulo siguiente
        capitulo = max((c for c in capitulos if c[0] < m.start()), key=lambda c: c[0])[1]
        salida.append({"id": f"Art. {m.group(1)}", "fuente": NOMBRE_REGLAMENTO,
                       "capitulo": f"Capítulo {capitulo}: {titulos.get(capitulo, '')}".strip(),
                       "texto": _limpiar(crudo)})
    return salida


def fragmentos_guia() -> list[dict]:
    paginas = [_limpiar(p) for p in _texto_pdf(GUIA, "plain")]
    secciones = [("Descripción del servicio", paginas[1:2]), ("Condiciones generales del servicio", paginas[2:3]),
                 ("Pasos para solicitar la cancelación en Banner", paginas[3:14]),
                 ("Ayuda y contacto", paginas[14:15])]
    return [{"id": f"Guía {i}", "fuente": NOMBRE_GUIA, "capitulo": titulo, "texto": "\n".join(ps)}
            for i, (titulo, ps) in enumerate(secciones, 1)]


def main() -> None:
    corpus = fragmentos_reglamento() + fragmentos_guia()
    _reparar_palabras(corpus)
    for f in corpus:
        if f["id"] == "Art. 68":
            assert _CUADRO_68[0] in f["texto"], "cambió el formato del cuadro del Art. 68"
            f["texto"] = f["texto"].replace(*_CUADRO_68)
    numeros = [int(f["id"].split()[1]) for f in corpus if f["id"].startswith("Art.")]
    assert numeros == list(range(1, len(numeros) + 1)), "faltan artículos o están desordenados"
    SALIDA.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    palabras = sum(len(f["texto"].split()) for f in corpus)
    print(f"{len(corpus)} fragmentos ({len(numeros)} artículos + {len(corpus) - len(numeros)} secciones de la guía), "
          f"{palabras} palabras → {SALIDA}")
