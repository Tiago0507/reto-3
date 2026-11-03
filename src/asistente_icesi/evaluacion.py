"""Evaluación comparativa: calidad, costo, latencia, robustez y coordinación.

Además de la rúbrica del juez se mide el *recall de artículos*: qué fracción de los
artículos que sustentan la respuesta esperada llegó a consultar el sistema.

Compara la misma pregunta por tres caminos del mismo grafo:
    A · agente único  (ruta="simple")   → baseline
    B · multiagente   (ruta="compleja")
    C · router        (ruta="auto")     → decide el router por pregunta

La calidad la califica un juez LLM de otra familia (reduce el sesgo de
autoevaluación) con la rúbrica del curso: cobertura 60 %, fidelidad 25 %,
claridad 15 %. El costo del juez se registra aparte.

    uv run asistente-icesi-evaluar --repeticiones 2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .configuracion import MODELOS, Contexto
from .grafo import construir_grafo
from .modelos import Trazador, crear_llm_estructurado, invocar_estructurado

# Preguntas reales de estudiantes; los puntos clave y los artículos esperados salen del texto
# vigente del reglamento (Res. 1.690/2026) y de la guía de cancelación de semestre.
DATASET: list[dict[str, Any]] = [
    {"pregunta": "¿Hasta qué semana puedo cancelar una materia y en qué casos me devuelven dinero?",
     "puntos": ["En las dos primeras semanas se cancela ante Admisiones y Registro con devolución de un porcentaje",
                "Desde la tercera hasta la duodécima semana la cancelación es extemporánea y no tiene reintegro de dinero",
                "La cancelación extemporánea causa un pago de derechos",
                "Una materia solo se puede cancelar dos veces"],
     "articulos": ["Art. 50", "Art. 53", "Art. 54"]},
    {"pregunta": "¿Con qué promedio entro en prueba académica?",
     "puntos": ["Promedio inferior a 3,3 al cabo del segundo semestre", "Promedio inferior a 3,4 en los semestres siguientes",
                "La prueba académica está vigente durante todo el semestre siguiente"],
     "articulos": ["Art. 67"]},
    {"pregunta": "Quedé en prueba académica, ¿qué consecuencias tiene para mi siguiente semestre?",
     "puntos": ["Solo puede matricular de forma autónoma hasta el 75 % de los créditos del semestre",
                "Para matricular más créditos necesita autorización del Director del Programa",
                "Debe organizar su matrícula de acuerdo con el Director del Programa",
                "Si estando en prueba no alcanza el promedio exigido, queda retirado"],
     "articulos": ["Art. 13", "Art. 66", "Art. 67", "Art. 68"]},
    {"pregunta": "¿Cuántas faltas de asistencia puedo tener antes de perder una materia?",
     "puntos": ["Si asiste a menos del 80 % de las horas de la materia recibe calificación de No aprobado",
                "Esa calificación no tiene valor numérico",
                "Solo aplica si el profesor lleva registro escrito de asistencia conocido por los estudiantes"],
     "articulos": ["Art. 70"]},
    {"pregunta": "¿Qué promedio necesito para graduarme con honores?",
     "puntos": ["Cum laude: promedio igual o superior a 4,3 e inferior a 4,5", "Magna cum laude: desde 4,5 e inferior a 4,8",
                "Summa cum laude: 4,8 o más",
                "Debe haber cursado en Icesi al menos el 75 % de los créditos y no haber sido sancionado disciplinariamente"],
     "articulos": ["Art. 83"]},
    {"pregunta": "No estoy de acuerdo con la nota de un parcial, ¿cómo y en qué plazo reclamo?",
     "puntos": ["El reclamo se presenta primero al profesor, por escrito",
                "Plazo de tres días hábiles desde que se comunica la calificación",
                "Si no queda satisfecho, acude al Director de su Programa",
                "Si procede, se designan dos nuevos calificadores y la nota es el promedio de ambos"],
     "articulos": ["Art. 77"]},
    {"pregunta": "¿En qué casos me retiran de la universidad por rendimiento académico?",
     "puntos": ["Según el número de materias perdidas en el período (por ejemplo, 3 o más si cursa 5 o más)",
                "Perder una materia en tercera repetición",
                "Perder una materia en segunda repetición con promedio del semestre inferior a 3,4",
                "Estar en prueba académica y no alcanzar el promedio exigido"],
     "articulos": ["Art. 68", "Art. 67"]},
    {"pregunta": "Me retiraron por bajo rendimiento académico, ¿puedo volver a la universidad y con qué condiciones?",
     "puntos": ["Puede pedir readmisión si no lleva más de cinco años desvinculado",
                "La solicitud es por escrito ante la oficina de Admisiones y Registro",
                "Solo puede ser readmitido máximo dos veces por bajo rendimiento",
                "Puede matricular hasta el 75 % de los créditos, con el Director del Programa",
                "Ingresa en prueba académica por readmisión"],
     "articulos": ["Art. 11", "Art. 12", "Art. 67"]},
    {"pregunta": "¿Puedo estudiar dos carreras al mismo tiempo en Icesi? ¿Qué requisitos hay y cuánto pago?",
     "puntos": ["No puede tener sanción disciplinaria ni estar en prueba académica al solicitarlo",
                "La solicitud se hace en Banner al cierre del semestre anterior y la deciden el Decano y el Director",
                "Paga la matrícula del programa de mayor valor",
                "Paga derechos de grado por cada programa"],
     "articulos": ["Art. 31", "Art. 32", "Art. 34"]},
    {"pregunta": "Quiero cancelar todo el semestre, ¿cómo lo hago y me devuelven la plata?",
     "puntos": ["En las dos primeras semanas puede cancelar sin aprobación de la facultad y con devolución de un porcentaje",
                "Después de la segunda semana requiere autorización del Director de Programa y del Decano",
                "Después de la segunda semana no hay devolución de dinero",
                "Se solicita en el autoservicio de Banner con el servicio Cancelación de semestre"],
     "articulos": ["Art. 50", "Art. 51", "Guía 2", "Guía 3"]},
    {"pregunta": "¿Qué necesito para estar en la Lista de Honor y para ganar una Beca de Honor?",
     "puntos": ["Promedio del período igual o mayor a 4,0", "Carga académica de al menos 15 créditos",
                "No haber perdido materias ni tener sanción o amonestación escrita en el semestre",
                "Las Becas de Honor se dan a los mejores promedios y devuelven un porcentaje de la matrícula del semestre anterior"],
     "articulos": ["Art. 89", "Art. 90", "Art. 91"]},
    {"pregunta": "Falté a un parcial porque estaba enfermo, ¿puedo presentarlo después?",
     "puntos": ["Puede presentar un examen supletorio por razones justificadas y comprobadas",
                "Debe solicitarlo por escrito al Director del Programa dentro de los dos días hábiles siguientes",
                "El supletorio tiene un valor que se paga en la tesorería"],
     "articulos": ["Art. 64"]},
    {"pregunta": "¿Qué requisitos debo cumplir para graduarme?",
     "puntos": ["Cumplir todos los requisitos del plan de estudios y los legales",
                "Estar a paz y salvo por todo concepto y pagar los derechos de grado",
                "Hacer proyecto de grado y semestre de práctica si el programa los exige",
                "No tener sanciones disciplinarias vigentes"],
     "articulos": ["Art. 84", "Art. 86"]},
    {"pregunta": "¿Cuánto cuesta la matrícula de un semestre de pregrado?",
     "puntos": ["El reglamento no fija el valor de la matrícula",
                "El valor lo establece anualmente la Rectoría mediante resolución"],
     "articulos": ["Art. 17"]},
]

CONFIGURACIONES = {
    "A · agente único": Contexto(ruta="simple"),
    "B · multiagente": Contexto(ruta="compleja"),
    "C · router": Contexto(ruta="auto"),
    "D · multiagente optimizado": Contexto(ruta="compleja", max_subtareas=2, max_rondas_critico=1, palabras_nota=100),
}

JUEZ = """Eres un evaluador estricto e imparcial de respuestas sobre el reglamento estudiantil de una universidad.
Compara la respuesta con los puntos clave esperados. Para cada punto clave, en el mismo orden, indica si la
respuesta lo contiene con el mismo sentido y el mismo dato (plazo, nota, porcentaje, instancia).
Califica fidelidad de 1 a 5 (5 = cada regla está citada y nada contradice los puntos; 1 = inventa normas o cifras)
y claridad de 1 a 5 (5 = responde directo a la duda del estudiante, breve y ordenada)."""


class Juicio(BaseModel):
    puntos_presentes: list[bool]
    fidelidad: int = Field(ge=1, le=5)
    claridad: int = Field(ge=1, le=5)
    justificacion: str


async def juzgar(item: dict, respuesta: str, trazador: Trazador) -> dict[str, float | None]:
    """Juez LLM con rúbrica: cobertura 60 %, fidelidad 25 %, claridad 15 %."""
    if not respuesta:
        return {"cobertura": 0.0, "fidelidad": 1, "claridad": 1, "calidad": 0.0}
    llm = crear_llm_estructurado(MODELOS["juez"], "juez", Juicio, max_tokens=500)
    entrada = (f"Pregunta: {item['pregunta']}\n<puntos>{json.dumps(item['puntos'], ensure_ascii=False)}</puntos>\n"
               f"<respuesta>{respuesta}</respuesta>")
    j, _ = await invocar_estructurado(llm, [SystemMessage(JUEZ), HumanMessage(entrada)], {"callbacks": [trazador]})
    if j is None:
        return {"cobertura": None, "fidelidad": None, "claridad": None, "calidad": None}
    n = len(item["puntos"])
    cobertura = sum((j.puntos_presentes + [False] * n)[:n]) / n
    calidad = 0.60 * cobertura + 0.25 * (j.fidelidad - 1) / 4 + 0.15 * (j.claridad - 1) / 4
    return {"cobertura": round(cobertura, 3), "fidelidad": j.fidelidad, "claridad": j.claridad,
            "calidad": round(calidad, 3)}


async def ejecutar_tarea(grafo, nombre: str, ctx: Contexto, item: dict, juez: Trazador) -> dict[str, Any]:
    """Una tarea = una pregunta con su propio trazador (tope de gasto y métricas)."""
    trazador = Trazador(tope_usd=ctx.tope_usd)
    config = {"callbacks": [trazador], "recursion_limit": 60,
              "metadata": {"tarea": uuid.uuid4().hex[:8], "config": nombre}}
    t0 = time.perf_counter()
    try:
        salida = await grafo.ainvoke({"messages": [HumanMessage(item["pregunta"])]}, config, context=ctx)
        respuesta, exito, error = salida["respuesta"], True, None
    except Exception as e:  # robustez: se mide la tasa de fallos, no se interrumpe el experimento
        salida, respuesta, exito, error = {}, "", False, repr(e)[:200]
    latencia = time.perf_counter() - t0
    docs = salida.get("documentos_consultados", [])
    esperados = set(item["articulos"])
    return {"config": nombre, "pregunta": item["pregunta"], "ruta": salida.get("ruta"), "exito": exito,
            "error": error, "latencia_s": round(latencia, 2), "costo_usd": trazador.costo_usd,
            "llamadas": len(trazador.llamadas),
            "tokens": sum(ll.tokens_entrada + ll.tokens_salida for ll in trazador.llamadas),
            "pct_coordinacion": round(trazador.pct_coordinacion(), 1),
            "pct_consultas_repetidas": round(100 * (len(docs) - len(set(docs))) / max(1, len(docs)), 1),
            "fidelidad_citas": (salida.get("verificacion") or {}).get("fidelidad"),
            "recall_articulos": round(len(esperados & set(docs)) / len(esperados), 3),
            "rondas": salida.get("rondas"), "incidentes": len(salida.get("incidentes", [])),
            **await juzgar(item, respuesta, juez), "respuesta": respuesta}


def _p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=20, method="inclusive")[-1] if len(xs) > 1 else xs[0]


def resumir(filas: list[dict]) -> list[dict[str, Any]]:
    resumen = []
    for nombre in dict.fromkeys(f["config"] for f in filas):
        g = [f for f in filas if f["config"] == nombre]
        calidades = [f["calidad"] for f in g if f["calidad"] is not None]
        costo = statistics.mean(f["costo_usd"] for f in g)
        resumen.append({
            "config": nombre,
            "calidad": round(statistics.mean(calidades), 3) if calidades else None,
            "éxito %": round(100 * sum(f["exito"] for f in g) / len(g)),
            "USD/tarea": round(costo, 5), "USD/1000": round(1000 * costo, 2),
            "p50 s": round(statistics.median(f["latencia_s"] for f in g), 1),
            "p95 s": round(_p95([f["latencia_s"] for f in g]), 1),
            "recall art.": round(statistics.mean(f["recall_articulos"] for f in g), 2),
            "llamadas": round(statistics.mean(f["llamadas"] for f in g), 1),
            "% coord.": round(statistics.mean(f["pct_coordinacion"] for f in g), 1),
            "% repetidas": round(statistics.mean(f["pct_consultas_repetidas"] for f in g), 1),
        })
    return resumen


def frontera_pareto(resumen: list[dict]) -> list[str]:
    """Configuraciones que ninguna otra supera a la vez en costo y calidad."""
    pts = [(r["config"], r["USD/tarea"], r["calidad"] or 0) for r in resumen]
    return [n for n, c, q in pts if not any(c2 <= c and q2 >= q and (c2 < c or q2 > q) for _, c2, q2 in pts)]


def _tabla(filas: list[dict]) -> str:
    cols = list(filas[0])
    ancho = {c: max(len(str(c)), *(len(str(f[c])) for f in filas)) for c in cols}
    linea = lambda f: "  ".join(str(f[c]).ljust(ancho[c]) for c in cols)  # noqa: E731
    return "\n".join([linea({c: c for c in cols}), "  ".join("-" * ancho[c] for c in cols), *map(linea, filas)])


async def evaluar(nombres: list[str], repeticiones: int, concurrencia: int) -> tuple[list[dict], Trazador]:
    grafo, juez = construir_grafo(), Trazador()
    semaforo = asyncio.Semaphore(concurrencia)  # evita límites de tasa del proveedor

    async def limitada(nombre: str, item: dict) -> dict:
        async with semaforo:
            return await ejecutar_tarea(grafo, nombre, CONFIGURACIONES[nombre], item, juez)

    filas = []
    for nombre in nombres:
        print(f"Evaluando {nombre}…")
        filas += await asyncio.gather(*[limitada(nombre, it) for it in DATASET for _ in range(repeticiones)])
    return filas, juez


def main() -> None:
    p = argparse.ArgumentParser(prog="asistente-icesi-evaluar", description=__doc__.split("\n\n")[0])
    p.add_argument("--configs", nargs="+", default=list(CONFIGURACIONES), choices=list(CONFIGURACIONES),
                   metavar="CONFIG", help=f"de: {', '.join(CONFIGURACIONES)}")
    p.add_argument("--repeticiones", type=int, default=1, help="las respuestas varían entre ejecuciones")
    p.add_argument("--concurrencia", type=int, default=3)
    p.add_argument("--salida", type=Path, default=Path("resultados"))
    args = p.parse_args()

    filas, juez = asyncio.run(evaluar(args.configs, args.repeticiones, args.concurrencia))
    resumen = resumir(filas)
    print("\n" + _tabla(resumen))
    print(f"\nFrontera de Pareto (costo vs. calidad): {', '.join(frontera_pareto(resumen))}")
    total = sum(f["costo_usd"] for f in filas)
    print(f"Costo del sistema: USD {total:.4f} · costo del juez: USD {juez.costo_usd:.4f}")

    args.salida.mkdir(exist_ok=True)
    archivo = args.salida / f"evaluacion_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with archivo.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Detalle por tarea: {archivo}")
