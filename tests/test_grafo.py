"""Pruebas del flujo del grafo con el modelo falso (sin red ni costo)."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from asistente_icesi.configuracion import Contexto
from asistente_icesi.grafo import construir_grafo
from asistente_icesi.modelos import PresupuestoExcedido, Trazador

PREGUNTA = {"messages": [HumanMessage("¿Hasta qué semana puedo cancelar una materia?")]}


async def test_ruta_simple_usa_un_solo_agente(escenario):
    salida = await construir_grafo().ainvoke(PREGUNTA, context=Contexto())
    assert salida["ruta"] == "simple"
    assert "[Art. 53]" in salida["respuesta"]
    assert "planificador" not in escenario.llamadas and "critico" not in escenario.llamadas
    assert salida["verificacion"]["fidelidad"] == 1.0


async def test_ruta_compleja_paraleliza_y_el_critico_itera(escenario):
    escenario.ruta, escenario.rechazos_critico = "compleja", 1
    trazador = Trazador()
    salida = await construir_grafo().ainvoke(PREGUNTA, {"callbacks": [trazador]}, context=Contexto(max_subtareas=2))
    assert salida["ruta"] == "compleja"
    assert escenario.llamadas.count("critico") == 2 and salida["rondas"] == 2
    assert escenario.llamadas.count("redactor") == 2
    assert {"investigador_1", "investigador_2"} <= {ll.agente for ll in trazador.llamadas}
    assert "investigador_3" not in {ll.agente for ll in trazador.llamadas}  # respeta max_subtareas


async def test_ruta_forzada_sin_historial_omite_el_router(escenario):
    salida = await construir_grafo().ainvoke(PREGUNTA, context=Contexto(ruta="compleja", max_rondas_critico=0))
    assert salida["ruta"] == "compleja" and "router" not in escenario.llamadas
    assert "critico" not in escenario.llamadas  # max_rondas_critico=0 apaga el evaluador


async def test_router_invalido_degrada_a_ruta_compleja(escenario):
    escenario.router_invalido = True
    salida = await construir_grafo().ainvoke(PREGUNTA, context=Contexto())
    assert salida["ruta"] == "compleja"
    assert any("router" in i for i in salida["incidentes"])


async def test_cita_invalida_nunca_se_aprueba(escenario):
    escenario.ruta, escenario.cita_invalida = "compleja", True
    salida = await construir_grafo().ainvoke(PREGUNTA, context=Contexto(max_rondas_critico=2))
    assert salida["verificacion"]["citas_invalidas"] == ["Art. 999"]
    assert salida["rondas"] == 2  # el crítico aprobó, pero la regla dura lo forzó a corregir


async def test_memoria_conversacional_por_hilo(escenario):
    grafo = construir_grafo(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    await grafo.ainvoke(PREGUNTA, config, context=Contexto())
    await grafo.ainvoke({"messages": [HumanMessage("¿Y cuántas veces se puede?")]}, config, context=Contexto())
    estado = await grafo.aget_state(config)
    assert len(estado.values["messages"]) == 4  # 2 preguntas + 2 respuestas
    assert len(estado.values["notas"]) == 1  # las notas se reinician en cada turno


async def test_revision_humana_con_interrupt(escenario):
    grafo = construir_grafo(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    ctx = Contexto(revision_humana=True)
    await grafo.ainvoke(PREGUNTA, config, context=ctx)
    estado = await grafo.aget_state(config)
    assert estado.next == ("revision_humana",) and estado.interrupts
    salida = await grafo.ainvoke(Command(resume="ok"), config, context=ctx)
    assert salida["respuesta"]


async def test_tope_de_gasto_detiene_la_tarea(escenario):
    escenario.ruta = "compleja"
    trazador = Trazador(tope_usd=1e-9)
    with pytest.raises(PresupuestoExcedido):
        await construir_grafo().ainvoke(PREGUNTA, {"callbacks": [trazador]}, context=Contexto())
