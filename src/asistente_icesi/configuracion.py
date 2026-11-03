"""Configuración: modelos por rol, precios de referencia y contexto de ejecución.

El `Contexto` es el *runtime context* de LangGraph (`context_schema`): datos
inmutables de una ejecución (modelos, límites, presupuesto) que los nodos leen
con `runtime.context`. No forma parte del estado ni se guarda en el checkpointer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]  # agente_icesi/

# Identificadores de OpenRouter. Decisión de diseño: barato para trabajar,
# otra familia para criticar (reduce el sesgo de autoevaluación).
MODELOS = {
    "barato": "deepseek/deepseek-v4.1-flash",
    "alterno": "openai/gpt-6-luna",
    "juez": "anthropic/claude-haiku-5.5",
    "fuerte": "google/gemini-3.7-flash",
}

# USD por millón de tokens (entrada, salida). Solo se usan si OpenRouter no
# devuelve el costo real de la llamada.
PRECIOS: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.028, 0.12),
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "google/gemini-3.7-flash": (0.75, 3.75),
}

# Modelos que no permiten apagar el razonamiento: se usa el esfuerzo mínimo.
RAZONAMIENTO_OBLIGATORIO = frozenset({"google/gemini-3.7-flash"})

MAX_SUBTAREAS = 5  # techo duro: se compila un investigador por índice

Ruta = Literal["simple", "compleja"]
Esfuerzo = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class Contexto:
    """Parámetros de una ejecución del grafo (inmutables durante la ejecución)."""

    # Modelo por rol
    modelo_router: str = MODELOS["barato"]
    modelo_planificador: str = MODELOS["barato"]
    modelo_trabajador: str = MODELOS["barato"]
    modelo_redactor: str = MODELOS["barato"]
    modelo_critico: str = MODELOS["alterno"]
    # Razonamiento interno: con "none" DeepSeek responde ~10 veces más barato en estas tareas.
    razonamiento: Esfuerzo = "none"

    # Control del flujo
    ruta: Literal["auto", "simple", "compleja"] = "auto"  # "auto" = decide el router
    max_subtareas: int = 3
    max_pasos_agente: int = 4  # ciclos pensar→actuar→observar por agente ReAct
    max_rondas_critico: int = 2
    palabras_nota: int = 150  # compresión de contexto de los investigadores
    palabras_respuesta: int = 220

    # Criterios de parada y supervisión
    tope_usd: float = 0.05  # freno de emergencia por tarea
    revision_humana: bool = False  # requiere checkpointer (usa `interrupt`)

    def __post_init__(self) -> None:
        if not 1 <= self.max_subtareas <= MAX_SUBTAREAS:
            raise ValueError(f"max_subtareas debe estar entre 1 y {MAX_SUBTAREAS}")
        if self.max_pasos_agente < 1 or self.max_rondas_critico < 0:
            raise ValueError("max_pasos_agente >= 1 y max_rondas_critico >= 0")
        if self.tope_usd <= 0:
            raise ValueError("tope_usd debe ser positivo")


def api_key() -> str:
    """Lee la clave de OpenRouter: variables de entorno, `.env` desde la carpeta actual
    hacia arriba o, si no, el `.env` de la raíz del proyecto (funciona desde cualquier carpeta)."""
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(RAIZ_PROYECTO / ".env")  # no pisa variables ya definidas
    clave = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not clave:
        raise ClaveFaltante(
            f"Falta OPENROUTER_API_KEY. Créala en {RAIZ_PROYECTO / '.env'} (ver .env.example)."
        )
    return clave


class ClaveFaltante(RuntimeError):
    """No se encontró la clave de OpenRouter."""
