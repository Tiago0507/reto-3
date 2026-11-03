"""Herramientas que pueden invocar los agentes.

Se declaran con `@tool`: nombre, docstring y tipos se convierten en el esquema
que ve el modelo. Las herramientas de consulta usan `response_format="content_and_artifact"`:
el modelo recibe el texto y el grafo recibe, aparte, los ids de los fragmentos
(el *artifact*), que sirven para verificar citas y medir trabajo repetido.
"""

from __future__ import annotations

import ast
import json
import operator

from langchain_core.tools import ToolException, tool

from .corpus import Fragmento, buscar, normalizar_id, por_id

MAX_PALABRAS_RESULTADO = 350  # los artículos largos se recortan en la búsqueda; leer_articulo da el texto completo


def _formato(f: Fragmento, recortar: bool) -> dict:
    palabras = f["texto"].split()
    texto = f["texto"]
    if recortar and len(palabras) > MAX_PALABRAS_RESULTADO:
        texto = " ".join(palabras[:MAX_PALABRAS_RESULTADO]) + f" […] (recortado: usa leer_articulo('{f['id']}'))"
    return {"id": f["id"], "capitulo": f["capitulo"], "texto": texto}


@tool(response_format="content_and_artifact")
def buscar_reglamento(consulta: str) -> tuple[str, list[str]]:
    """Busca en el reglamento estudiantil de pregrado de la Universidad Icesi (2026) y en la guía
    de cancelación de semestre. Devuelve hasta 4 fragmentos con id (por ejemplo "Art. 53"), capítulo y texto.

    Usa consultas cortas con los términos del reglamento, por ejemplo "cancelación de materias",
    "prueba académica promedio" o "grado con honores".
    """
    fragmentos = buscar(consulta)
    if not fragmentos:
        return "Sin resultados. Prueba con otros términos.", []
    return json.dumps([_formato(f, True) for f in fragmentos], ensure_ascii=False), [f["id"] for f in fragmentos]


@tool(response_format="content_and_artifact")
def leer_articulo(identificador: str) -> tuple[str, list[str]]:
    """Devuelve el texto completo de un artículo o sección, por ejemplo "Art. 68", "68" o "Guía 2".

    Úsala cuando un artículo remite a otro ("según el Artículo 67") o cuando un resultado salió recortado.
    """
    clave = normalizar_id(identificador)
    if clave is None or clave not in por_id():
        raise ToolException(f"No existe '{identificador}'. Hay artículos del 1 al 110 y secciones Guía 1 a Guía 4.")
    return json.dumps(_formato(por_id()[clave], False), ensure_ascii=False), [clave]


_OPERADORES = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod,
}


def _evaluar(nodo: ast.AST) -> float:
    match nodo:
        case ast.Expression(body=cuerpo):
            return _evaluar(cuerpo)
        case ast.Constant(value=int() | float() as v) if not isinstance(v, bool):
            return v
        case ast.BinOp(left=izq, op=op, right=der) if type(op) in _OPERADORES:
            if isinstance(op, ast.Pow) and abs(_evaluar(der)) > 100:
                raise ToolException("Exponente demasiado grande.")
            return _OPERADORES[type(op)](_evaluar(izq), _evaluar(der))
        case ast.UnaryOp(op=op, operand=x) if type(op) in _OPERADORES:
            return _OPERADORES[type(op)](_evaluar(x))
    raise ToolException("Solo se permiten números y los operadores + - * / ** % y paréntesis.")


@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión aritmética, por ejemplo "(4.2*3 + 3.8*4) / 7" para un promedio ponderado
    o "0.8 * 64" para el mínimo de horas de asistencia. No admite variables ni funciones.
    """
    try:
        resultado = _evaluar(ast.parse(expresion.replace(",", "."), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolException(f"Expresión inválida: {e}") from e
    return f"{resultado:.6g}"


HERRAMIENTAS = [buscar_reglamento, leer_articulo, calcular]
