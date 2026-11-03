"""Verificador determinista de citas (sin LLM, costo cero).

Revisa, frase por frase, que cada cita [Art. N] o [Guía N] exista y haya sido
consultada, y que cada cifra (notas, porcentajes, plazos) aparezca en alguno de
los fragmentos citados en esa frase. Las citas inválidas son graves (el crítico
no puede aprobar); las cifras sin respaldo son advertencias, porque pueden ser
cálculos legítimos (por ejemplo, el 80 % de 48 horas = 38,4 horas).
"""

from __future__ import annotations

import re

from .corpus import normalizar_id, por_id
from .estado import Verificacion

# Acepta [Art. 53], [Art 53], [Artículo 53], [Guía 2] y varias citas en un corchete: [Art. 50, Art. 53]
_CORCHETE = re.compile(r"\[([^\[\]]*?(?:Art|Guía|Guia)[^\[\]]*?)\]", re.I)
_REFERENCIA = re.compile(r"(?:art(?:[íi]culo)?s?\.?\s*\d+|gu[íi]a\s*\d+)", re.I)
_CIFRA = re.compile(r"\d+(?:[.,]\d+)*")
_FRASE = re.compile(r"(?<=[.!?])(?<![Aa]rt\.)\s+|\n+")  # "Art. 53" no termina la frase
_PALABRAS_NUMERO = {"uno": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7",
                    "ocho": "8", "nueve": "9", "diez": "10", "doce": "12", "quince": "15", "dieciseis": "16"}


def citas(frase: str) -> list[str]:
    """Ids normalizados de las citas entre corchetes; los no reconocidos se devuelven tal cual."""
    salida = []
    for contenido in _CORCHETE.findall(frase):
        for ref in _REFERENCIA.findall(contenido):
            salida.append(normalizar_id(ref) or ref)
    return salida


def _cifras(texto: str) -> set[str]:
    """Cifras normalizadas ('3,4' → '3.4'); se ignoran las referencias a artículos, los años y los dígitos sueltos."""
    texto = _REFERENCIA.sub(" ", _CORCHETE.sub(" ", texto))
    salida = set()
    for c in _CIFRA.findall(texto):
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", c):  # separador de miles
            c = c.replace(".", "")
        c = c.replace(",", ".")
        if len(c) == 1 or re.fullmatch(r"(19|20)\d\d", c):
            continue
        salida.add(c)
    return salida


def _cifras_fuente(texto: str) -> set[str]:
    """Cifras de un artículo, incluidas las escritas en letras ('dos semanas' → '2')."""
    salida = _cifras(texto) | {c.replace(",", ".") for c in _CIFRA.findall(texto)}
    for palabra, numero in _PALABRAS_NUMERO.items():
        if re.search(rf"\b{palabra}\b", texto, re.I):
            salida.add(numero)
    return salida


def verificar_citas(respuesta: str, consultados: set[str]) -> Verificacion:
    citas_invalidas: list[str] = []
    sin_respaldo: list[str] = []
    sin_cita: list[str] = []
    total = respaldadas = 0
    fragmentos = por_id()

    for frase in filter(None, (f.strip() for f in _FRASE.split(respuesta))):
        citadas = citas(frase)
        for c in citadas:
            if c not in fragmentos or c not in consultados:
                citas_invalidas.append(c)
        cifras = _cifras(frase)
        if not cifras:
            continue
        if not citadas:
            sin_cita.append(frase[:120])
            total += len(cifras)
            continue
        fuente = set().union(*(_cifras_fuente(fragmentos[c]["texto"]) for c in citadas if c in fragmentos))
        for c in cifras:
            total += 1
            if c in fuente:
                respaldadas += 1
            else:
                sin_respaldo.append(f"{c} {citadas}")

    return {"citas_invalidas": sorted(set(citas_invalidas)), "cifras_sin_respaldo": sin_respaldo,
            "cifras_sin_cita": sin_cita, "fidelidad": round(respaldadas / total, 3) if total else 1.0}


def describir(v: Verificacion) -> str:
    """Resumen legible para el crítico y el redactor."""
    partes = []
    if v["citas_invalidas"]:
        partes.append(f"GRAVE: citas a artículos inexistentes o no consultados: {', '.join(v['citas_invalidas'])}.")
    if v["cifras_sin_respaldo"]:
        partes.append("Cifras que no aparecen en el artículo citado (aceptable solo si es un cálculo explícito): "
                      + "; ".join(v["cifras_sin_respaldo"][:8]) + ".")
    if v["cifras_sin_cita"]:
        partes.append("Frases con cifras sin cita: " + " | ".join(v["cifras_sin_cita"][:5]) + ".")
    if not partes:
        partes.append("Sin hallazgos: todas las cifras están citadas y respaldadas.")
    return f"Fidelidad de citas: {v['fidelidad']:.0%}. " + " ".join(partes)
