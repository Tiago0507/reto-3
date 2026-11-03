"""Grafo del asistente del reglamento estudiantil de Icesi: combina cuatro patrones multiagente.

    START → enrutar ──simple───▶ responder_directo (ReAct) ──┐
               │                                             ▼
               └─compleja─▶ planificar ─Send×n─▶ investigar ─▶ redactar ─▶ verificar
                                         (en paralelo)          ▲             │
                                                                │   compleja  ▼
                                                                └─rechaza── criticar
                                                                              │ aprueba
                                          finalizar ◀── revision_humana ◀─────┘
                                              │          (interrupt opcional)
                                             END

| Patrón                     | Pieza de LangGraph                                         |
|----------------------------|------------------------------------------------------------|
| Router                     | `enrutar` devuelve `Command(goto=...)`                     |
| Orquestador–trabajadores   | `planificar` escribe subtareas; cada investigador es un    |
|                            | subgrafo ReAct con su propio contexto                      |
| Paralelización (map-reduce)| `Send` por subtarea + reductor `acumular` sobre `notas`    |
| Evaluador–optimizador      | `criticar` vuelve a `redactar` hasta aprobar o agotar      |
|                            | `max_rondas_critico`                                       |
| Humano en el bucle         | `interrupt()` en `revision_humana` + checkpointer          |
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send, interrupt

from . import prompts
from .agentes import analista, documentos_de, investigador
from .configuracion import Contexto
from .esquemas import Enrutamiento, Evaluacion, Plan
from .estado import Entrada, EntradaInvestigador, Estado, Salida, reinicio_de_turno
from .modelos import crear_llm, crear_llm_estructurado, invocar_estructurado, texto
from .politicas import REINTENTO_LLM
from .verificacion import describir, verificar_citas

HISTORIAL_ROUTER = 6  # mensajes previos que ve el router para resolver referencias ("¿y esa opción?")


def _historial(mensajes: list[AnyMessage]) -> str:
    previos = mensajes[:-1][-HISTORIAL_ROUTER:]
    rol = {True: "Estudiante", False: "Asistente"}
    return "\n".join(f"{rol[isinstance(m, HumanMessage)]}: {texto(m)[:400]}" for m in previos)


def _contexto_notas(state: Estado) -> str:
    notas = "\n\n".join(f"### {n['subtarea']}\n{n['contenido']}" for n in state.get("notas", []))
    return f"Pregunta: {state['pregunta']}\n\nNotas:\n{notas}"


# --- Nodos ------------------------------------------------------------------

async def enrutar(state: Estado, runtime: Runtime[Contexto]) -> Command[Literal["responder_directo", "planificar"]]:
    """Router: reescribe la pregunta con el historial y decide la ruta."""
    ctx = runtime.context
    ultima = texto(state["messages"][-1])
    historial = _historial(state["messages"])
    actualizacion = reinicio_de_turno()
    incidentes: list[str] = []

    if ctx.ruta != "auto" and not historial:  # ruta forzada y nada que reescribir: sin llamada al LLM
        pregunta, ruta = ultima, ctx.ruta
    else:
        entrada = f"Historial:\n{historial}\n\nÚltima pregunta: {ultima}" if historial else ultima
        llm = crear_llm_estructurado(ctx.modelo_router, "router", Enrutamiento, max_tokens=300,
                                     razonamiento=ctx.razonamiento)
        decision, error = await invocar_estructurado(llm, [SystemMessage(prompts.ROUTER), HumanMessage(entrada)])
        if decision is None:  # degrada con elegancia: la ruta completa es la más segura
            incidentes.append(f"router: salida inválida ({error}); se usa ruta compleja")
            pregunta, ruta = ultima, "compleja"
        else:
            pregunta, ruta = decision.pregunta or ultima, decision.complejidad
        if ctx.ruta != "auto":
            ruta = ctx.ruta

    actualizacion |= {"pregunta": pregunta, "ruta": ruta, "incidentes": incidentes}
    return Command(goto="responder_directo" if ruta == "simple" else "planificar", update=actualizacion)


async def responder_directo(state: Estado, runtime: Runtime[Contexto]) -> dict:
    """Ruta simple: un solo agente ReAct con todas las herramientas (el baseline)."""
    salida = await analista().ainvoke({"messages": [HumanMessage(state["pregunta"])]}, context=runtime.context)
    respuesta, docs = texto(salida["messages"][-1]), documentos_de(salida["messages"])
    return {"borrador": respuesta, "documentos_consultados": docs,
            "notas": [{"subtarea": "Investigación directa", "contenido": respuesta, "documentos": docs}]}


async def planificar(state: Estado, runtime: Runtime[Contexto]) -> dict:
    """Orquestador: divide la pregunta en subtareas independientes."""
    ctx = runtime.context
    llm = crear_llm_estructurado(ctx.modelo_planificador, "planificador", Plan, max_tokens=400,
                                 razonamiento=ctx.razonamiento)
    plan, error = await invocar_estructurado(
        llm, [SystemMessage(prompts.PLANIFICADOR.format(n=ctx.max_subtareas)), HumanMessage(state["pregunta"])])
    if plan is None:
        return {"subtareas": [state["pregunta"]],
                "incidentes": [f"planificador: plan inválido ({error}); una sola subtarea"]}
    return {"subtareas": plan.subtareas[: ctx.max_subtareas]}


def repartir(state: Estado) -> list[Send]:
    """Map: un `Send` por subtarea; LangGraph ejecuta los investigadores en paralelo."""
    return [Send("investigar", {"pregunta": state["pregunta"], "subtarea": st, "indice": i})
            for i, st in enumerate(state["subtareas"], 1)]


async def investigar(entrada: EntradaInvestigador, runtime: Runtime[Contexto]) -> dict:
    """Trabajador: ve solo su subtarea (contexto limpio). Reduce: su nota se agrega a `notas`."""
    salida = await investigador(entrada["indice"]).ainvoke(
        {"messages": [HumanMessage(f"Pregunta general: {entrada['pregunta']}\nTu subtarea: {entrada['subtarea']}")]},
        context=runtime.context)
    docs = documentos_de(salida["messages"])
    nota = {"subtarea": entrada["subtarea"], "contenido": texto(salida["messages"][-1]), "documentos": docs}
    return {"notas": [nota], "documentos_consultados": docs}


async def redactar(state: Estado, runtime: Runtime[Contexto]) -> dict:
    """Redactor: sintetiza las notas; si hay comentarios, corrige el borrador anterior."""
    ctx = runtime.context
    entrada = _contexto_notas(state)
    if state.get("comentarios"):
        entrada += f"\n\nBorrador anterior:\n{state['borrador']}\n\nComentarios de revisión:\n{state['comentarios']}"
    r = await crear_llm(ctx.modelo_redactor, "redactor", max_tokens=700, razonamiento=ctx.razonamiento).ainvoke(
        [SystemMessage(prompts.REDACTOR.format(palabras=ctx.palabras_respuesta)), HumanMessage(entrada)])
    return {"borrador": texto(r)}


def verificar(state: Estado, runtime: Runtime[Contexto]) -> Command[Literal["criticar", "redactar", "revision_humana"]]:
    """Verificador determinista de citas; decide si hace falta el crítico o una corrección."""
    v = verificar_citas(state["borrador"], set(state.get("documentos_consultados", [])))
    rondas, maximo = state.get("rondas", 0), runtime.context.max_rondas_critico
    if rondas >= maximo:
        aviso = "se agotaron las rondas de revisión; se entrega el último borrador"
        nuevo = [] if maximo == 0 or aviso in state.get("incidentes", []) else [aviso]
        return Command(goto="revision_humana", update={"verificacion": v, "incidentes": nuevo})
    if state["ruta"] == "compleja":
        return Command(goto="criticar", update={"verificacion": v})
    if v["citas_invalidas"]:  # ruta simple: solo se corrige si hay errores graves
        return Command(goto="redactar", update={"verificacion": v, "rondas": rondas + 1, "comentarios": describir(v)})
    return Command(goto="revision_humana", update={"verificacion": v})


async def criticar(state: Estado, runtime: Runtime[Contexto]) -> Command[Literal["redactar", "revision_humana"]]:
    """Evaluador: aprueba o devuelve el borrador al redactor con comentarios."""
    ctx = runtime.context
    v = state["verificacion"]
    entrada = (f"{_contexto_notas(state)}\n\nRespuesta propuesta:\n{state['borrador']}\n\n"
               f"Verificador automático: {describir(v)}")
    llm = crear_llm_estructurado(ctx.modelo_critico, "critico", Evaluacion, max_tokens=400,
                                 razonamiento=ctx.razonamiento)
    ev, error = await invocar_estructurado(llm, [SystemMessage(prompts.CRITICO), HumanMessage(entrada)])
    rondas = state.get("rondas", 0) + 1
    if ev is None:
        return Command(goto="revision_humana", update={
            "rondas": rondas, "aprobado": True,
            "incidentes": [f"critico: evaluación inválida ({error}); se aprueba"]})
    aprobado = ev.aprobado and not v["citas_invalidas"]  # las citas inválidas nunca se aprueban
    if aprobado:
        return Command(goto="revision_humana", update={"rondas": rondas, "aprobado": True, "comentarios": ""})
    comentarios = ev.comentarios or "Corrige los hallazgos del verificador."
    if v["citas_invalidas"]:
        comentarios += "\n" + describir(v)
    return Command(goto="redactar", update={"rondas": rondas, "aprobado": False, "comentarios": comentarios})


def revision_humana(state: Estado, runtime: Runtime[Contexto]) -> Command[Literal["redactar", "finalizar"]]:
    """Humano en el bucle (opcional): pausa el grafo y espera aprobación o comentarios."""
    if not runtime.context.revision_humana:
        return Command(goto="finalizar")
    decision = interrupt({"borrador": state["borrador"], "verificacion": state["verificacion"],
                          "instrucciones": "Escribe 'ok' para aprobar o tus comentarios para corregir."})
    if str(decision).strip().lower() in {"", "ok", "aprobar", "si", "sí", "s"}:
        return Command(goto="finalizar")
    return Command(goto="redactar", update={"comentarios": f"Comentarios del revisor humano: {decision}"})


def finalizar(state: Estado) -> dict:
    """Publica la respuesta y la agrega a la memoria de la conversación."""
    return {"respuesta": state["borrador"], "messages": [AIMessage(state["borrador"], name="asistente")]}


# --- Construcción -----------------------------------------------------------

def construir_grafo(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(Estado, context_schema=Contexto, input_schema=Entrada, output_schema=Salida)
    g.add_node("enrutar", enrutar, retry_policy=REINTENTO_LLM)
    g.add_node("responder_directo", responder_directo)
    g.add_node("planificar", planificar, retry_policy=REINTENTO_LLM)
    g.add_node("investigar", investigar)
    g.add_node("redactar", redactar, retry_policy=REINTENTO_LLM)
    g.add_node("verificar", verificar)
    g.add_node("criticar", criticar, retry_policy=REINTENTO_LLM)
    g.add_node("revision_humana", revision_humana)
    g.add_node("finalizar", finalizar)

    g.add_edge(START, "enrutar")
    g.add_conditional_edges("planificar", repartir, ["investigar"])
    g.add_edge("investigar", "redactar")  # espera a que terminen todos los investigadores
    g.add_edge("responder_directo", "verificar")
    g.add_edge("redactar", "verificar")
    g.add_edge("finalizar", END)
    return g.compile(checkpointer=checkpointer, name="asistente_icesi")


# Grafo sin checkpointer para LangGraph Server / Studio (la plataforma pone la persistencia).
grafo = construir_grafo()
