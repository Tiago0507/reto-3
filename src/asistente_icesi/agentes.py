"""Agente ReAct como subgrafo: pensar → actuar → observar.

Un agente es modelo + instrucciones + herramientas + memoria + criterio de parada:
- `modelo`: el LLM decide si responde o pide herramientas.
- `herramientas`: `ToolNode` las ejecuta (en paralelo si el modelo pide varias).
- `tools_condition` cierra el ciclo; la memoria es la lista de mensajes.
- Parada: al llegar a `max_pasos_agente` el modelo debe responder sin herramientas.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from . import prompts
from .configuracion import Contexto
from .herramientas import HERRAMIENTAS
from .modelos import crear_llm
from .politicas import REINTENTO_LLM


def construir_agente_react(nombre: str, instrucciones: Callable[[Contexto], str]) -> CompiledStateGraph:
    async def modelo(state: MessagesState, runtime: Runtime[Contexto]) -> dict:
        ctx = runtime.context
        mensajes: list[AnyMessage] = [SystemMessage(instrucciones(ctx)), *state["messages"]]
        pasos = sum(isinstance(m, AIMessage) for m in state["messages"])
        if pasos >= ctx.max_pasos_agente:  # criterio de parada
            llm = crear_llm(ctx.modelo_trabajador, nombre, herramientas=HERRAMIENTAS, tool_choice="none",
                            razonamiento=ctx.razonamiento)
            mensajes.append(HumanMessage(prompts.CIERRE_AGENTE))
        else:
            llm = crear_llm(ctx.modelo_trabajador, nombre, herramientas=HERRAMIENTAS, razonamiento=ctx.razonamiento)
        return {"messages": [await llm.ainvoke(mensajes)]}

    g = StateGraph(MessagesState, context_schema=Contexto)
    g.add_node("modelo", modelo, retry_policy=REINTENTO_LLM)
    g.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))
    g.add_edge(START, "modelo")
    g.add_conditional_edges("modelo", tools_condition, {"tools": "herramientas", END: END})
    g.add_edge("herramientas", "modelo")
    return g.compile(name=nombre)


def documentos_de(mensajes: list[AnyMessage]) -> list[str]:
    """Ids de artículos devueltos por buscar_reglamento y leer_articulo (vienen en el *artifact*)."""
    return [d for m in mensajes if isinstance(m, ToolMessage) and isinstance(m.artifact, list) for d in m.artifact]


@cache
def analista() -> CompiledStateGraph:
    return construir_agente_react("analista", lambda c: prompts.ANALISTA.format(palabras=c.palabras_respuesta))


@cache
def investigador(indice: int) -> CompiledStateGraph:
    """Un subgrafo por índice: así cada investigador tiene su propio nombre en las trazas."""
    return construir_agente_react(f"investigador_{indice}",
                                  lambda c: prompts.INVESTIGADOR.format(palabras=c.palabras_nota))
