"""Estado del grafo: el contrato entre agentes.

Cada nodo lee solo las claves que necesita y devuelve solo las que cambia.
Las listas que llenan varios nodos en paralelo usan un *reductor*.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from .configuracion import Ruta

REINICIAR = None  # valor que, devuelto por un nodo, vacía una lista acumulativa


def acumular(actual: list | None, nuevo: list | None) -> list:
    """Reductor: concatena, o reinicia si recibe `REINICIAR` (cada turno nuevo)."""
    if nuevo is REINICIAR:
        return []
    return [*(actual or []), *nuevo]


class Nota(TypedDict):
    subtarea: str
    contenido: str
    documentos: list[str]


class Verificacion(TypedDict):
    citas_invalidas: list[str]  # ids citados que no existen o no se consultaron (grave)
    cifras_sin_respaldo: list[str]  # cifras que no aparecen en los documentos citados (advertencia)
    cifras_sin_cita: list[str]  # frases con cifras y sin cita (advertencia)
    fidelidad: float  # fracción de cifras respaldadas por el documento citado


class Estado(MessagesState):
    """Estado completo (interno) del asistente. `messages` es la memoria de la conversación."""

    pregunta: str
    ruta: Ruta
    subtareas: list[str]
    notas: Annotated[list[Nota], acumular]
    documentos_consultados: Annotated[list[str], acumular]
    borrador: str
    comentarios: str
    verificacion: Verificacion
    aprobado: bool
    rondas: int
    respuesta: str
    incidentes: Annotated[list[str], acumular]  # degradaciones registradas (robustez)


class Entrada(TypedDict):
    """Lo que recibe el grafo: el nuevo mensaje del usuario."""

    messages: Annotated[list[AnyMessage], add_messages]


class Salida(TypedDict):
    """Lo que entrega el grafo."""

    respuesta: str
    ruta: Ruta
    documentos_consultados: list[str]
    verificacion: Verificacion
    rondas: int
    incidentes: list[str]


class EntradaInvestigador(TypedDict):
    """Estado privado de un investigador: solo su subtarea (contexto limpio)."""

    pregunta: str
    subtarea: str
    indice: int


def reinicio_de_turno() -> dict[str, Any]:
    """Valores con que arranca cada turno; la conversación (`messages`) se conserva."""
    return {"subtareas": [], "notas": REINICIAR, "documentos_consultados": REINICIAR,
            "incidentes": REINICIAR, "borrador": "", "comentarios": "", "aprobado": False,
            "rondas": 0, "respuesta": "",
            "verificacion": {"citas_invalidas": [], "cifras_sin_respaldo": [], "cifras_sin_cita": [],
                             "fidelidad": 1.0}}
