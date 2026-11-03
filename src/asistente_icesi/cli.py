"""Interfaz de línea de comandos.

    asistente-icesi "¿Hasta qué semana puedo cancelar una materia?"   # una pregunta
    asistente-icesi                                                   # conversación (memoria por hilo)
    asistente-icesi --ruta compleja --revisar --trazas "..."          # forzar ruta, humano en el bucle, costos
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .configuracion import MODELOS, ClaveFaltante, Contexto, api_key
from .grafo import construir_grafo
from .modelos import PresupuestoExcedido, Trazador

_ETIQUETAS = {
    "enrutar": "router", "responder_directo": "agente único (ReAct)", "planificar": "orquestador",
    "investigar": "investigador", "redactar": "redactor", "verificar": "verificador de citas",
    "criticar": "crítico", "revision_humana": "revisión", "finalizar": "final",
}


def _ids(ids: list[str]) -> str:
    """Ids únicos en orden natural: Art. 9, Art. 50, Guía 2."""
    return ", ".join(sorted(set(ids), key=lambda i: (i.startswith("Guía"), int(i.split()[-1])))) or "nada"


def _describir_evento(nodo: str, cambios: dict[str, Any] | None) -> str:
    c = cambios or {}
    detalle = ""
    if nodo == "enrutar":
        detalle = f"ruta={c.get('ruta')} · «{c.get('pregunta', '')[:70]}»"
    elif nodo == "planificar":
        detalle = f"{len(c.get('subtareas', []))} subtareas"
    elif nodo == "investigar" and c.get("notas"):
        n = c["notas"][0]
        detalle = f"{n['subtarea'][:60]} · {_ids(n['documentos'])}"
    elif nodo == "verificar" and c.get("verificacion"):
        detalle = f"fidelidad {c['verificacion']['fidelidad']:.0%}"
    elif nodo == "criticar":
        detalle = f"ronda {c.get('rondas')} · {'aprueba' if c.get('aprobado') else 'pide cambios'}"
    return f"  · {_ETIQUETAS.get(nodo, nodo):<22} {detalle}"


async def responder(grafo, entrada: Any, config: dict, ctx: Contexto, verbose: bool) -> dict | None:
    """Ejecuta un turno con streaming; atiende los `interrupt` de revisión humana."""
    while True:
        interrupcion = None
        async for evento in grafo.astream(entrada, config, context=ctx, stream_mode="updates"):
            for nodo, cambios in evento.items():
                if nodo == "__interrupt__":
                    interrupcion = cambios[0].value
                elif verbose:
                    print(_describir_evento(nodo, cambios), file=sys.stderr)
        if interrupcion is None:
            return (await grafo.aget_state(config)).values
        print(f"\n--- Borrador para revisión ---\n{interrupcion['borrador']}\n", file=sys.stderr)
        entrada = Command(resume=input(f"{interrupcion['instrucciones']}\n> "))


def _imprimir_trazas(t: Trazador) -> None:
    print(f"\n{'agente':<16}{'llamadas':>9}{'tok. entrada':>14}{'tok. salida':>13}{'USD':>11}", file=sys.stderr)
    for agente, f in sorted(t.por_agente().items(), key=lambda kv: -kv[1]["costo_usd"]):
        print(f"{agente:<16}{f['llamadas']:>9.0f}{f['tokens_entrada']:>14.0f}{f['tokens_salida']:>13.0f}"
              f"{f['costo_usd']:>11.5f}", file=sys.stderr)
    print(f"{'TOTAL':<16}{len(t.llamadas):>9}{'':>27}{t.costo_usd:>11.5f}"
          f"   ({t.pct_coordinacion():.0f} % en coordinación)", file=sys.stderr)


async def _turno(grafo, pregunta: str, hilo: str, ctx: Contexto, args) -> None:
    trazador = Trazador(tope_usd=ctx.tope_usd)  # un trazador por tarea: tope y métricas propias
    config = {"configurable": {"thread_id": hilo}, "callbacks": [trazador], "recursion_limit": 60,
              "metadata": {"tarea": uuid.uuid4().hex[:8]}}
    try:
        estado = await responder(grafo, {"messages": [HumanMessage(pregunta)]}, config, ctx, not args.silencioso)
    except PresupuestoExcedido as e:
        print(f"\n[detenido] {e}", file=sys.stderr)
        return
    except ClaveFaltante:
        raise
    except Exception as e:  # un error de red o del proveedor no debe cerrar la conversación
        print(f"\n[error] No se pudo responder: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return
    print(f"\n{estado['respuesta']}\n")
    if not args.silencioso:
        v = estado["verificacion"]
        print(f"[ruta {estado['ruta']} · consultó {_ids(estado['documentos_consultados'])} · "
              f"fidelidad de citas {v['fidelidad']:.0%} · rondas {estado['rondas']} · "
              f"USD {trazador.costo_usd:.5f}]", file=sys.stderr)
        for inc in estado.get("incidentes", []):
            print(f"[incidente] {inc}", file=sys.stderr)
    if args.trazas:
        _imprimir_trazas(trazador)


async def _principal(args) -> None:
    ctx = Contexto(ruta=args.ruta, revision_humana=args.revisar, tope_usd=args.tope_usd,
                   **({"modelo_redactor": MODELOS["fuerte"], "modelo_planificador": MODELOS["fuerte"]}
                      if args.mixto else {}))
    grafo = construir_grafo(checkpointer=InMemorySaver())  # memoria de la conversación por hilo
    hilo = uuid.uuid4().hex
    if args.pregunta:
        await _turno(grafo, " ".join(args.pregunta), hilo, ctx, args)
        return
    print("Asistente del reglamento estudiantil de pregrado de la Universidad Icesi.\n"
          "Escribe tu duda (vacío o 'salir' para terminar).", file=sys.stderr)
    while (pregunta := input("\n> ").strip()) and pregunta.lower() not in {"salir", "exit", "q"}:
        await _turno(grafo, pregunta, hilo, ctx, args)


def main() -> None:
    p = argparse.ArgumentParser(prog="asistente-icesi",
                                description="Asistente multiagente del reglamento estudiantil de pregrado de Icesi.")
    p.add_argument("pregunta", nargs="*", help="pregunta; sin ella se abre una conversación")
    p.add_argument("--ruta", choices=["auto", "simple", "compleja"], default="auto",
                   help="auto = decide el router (por defecto)")
    p.add_argument("--revisar", action="store_true", help="humano en el bucle antes de entregar")
    p.add_argument("--mixto", action="store_true", help="planificador y redactor con el modelo fuerte")
    p.add_argument("--tope-usd", type=float, default=0.05, help="tope de gasto por pregunta")
    p.add_argument("--trazas", action="store_true", help="tabla de costo por agente")
    p.add_argument("--silencioso", action="store_true", help="solo la respuesta")
    args = p.parse_args()
    try:
        api_key()  # falla pronto y con un mensaje claro si no hay clave
        asyncio.run(_principal(args))
    except ClaveFaltante as e:
        sys.exit(f"Error: {e}")
    except (KeyboardInterrupt, EOFError):
        print(file=sys.stderr)
