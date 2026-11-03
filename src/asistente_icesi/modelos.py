"""Fábrica de modelos de chat (OpenRouter) y trazas de costo con callbacks.

- `ChatOpenAI` funciona con OpenRouter porque su API es compatible con la de OpenAI.
- Cada llamada se etiqueta con el agente que la hace (`metadata`), así el
  `Trazador` puede atribuir costo, tokens y latencia por agente.
- El `Trazador` aplica el tope de gasto por tarea: si ya se superó, la siguiente
  llamada a un modelo se cancela con `PresupuestoExcedido`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel

from .configuracion import OPENROUTER_BASE_URL, PRECIOS, RAZONAMIENTO_OBLIGATORIO, Esfuerzo, api_key


class PresupuestoExcedido(RuntimeError):
    """La tarea superó su tope de gasto en USD."""


MARGEN_RAZONAMIENTO = 2000  # tokens extra de salida cuando el modelo razona antes de responder


@lru_cache(maxsize=64)
def _chat(modelo: str, max_tokens: int, razonamiento: Esfuerzo) -> BaseChatModel:
    if razonamiento == "none" and modelo in RAZONAMIENTO_OBLIGATORIO:
        razonamiento = "low"
    if razonamiento != "none":
        max_tokens += MARGEN_RAZONAMIENTO  # evita que el razonamiento agote el presupuesto de la respuesta
    return ChatOpenAI(
        model=modelo,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key(),
        temperature=0.2,
        max_tokens=max_tokens,
        max_retries=2,  # reintentos HTTP con espera exponencial
        timeout=90,
        extra_body={"usage": {"include": True},  # OpenRouter devuelve el costo real
                    "reasoning": {"effort": razonamiento, "exclude": True}},
        default_headers={"X-Title": "Asistente reglamento Icesi - LangGraph"},
    )


def _etiquetar(r: Runnable, agente: str, modelo: str) -> Runnable:
    return r.with_config(run_name=agente, tags=[f"agente:{agente}"],
                         metadata={"agente": agente, "modelo": modelo})


def crear_llm(modelo: str, agente: str, *, herramientas: list[BaseTool] | None = None,
              tool_choice: str | None = None, max_tokens: int = 800, razonamiento: Esfuerzo = "none") -> Runnable:
    llm: Runnable = _chat(modelo, max_tokens, razonamiento)
    if herramientas:
        llm = llm.bind_tools(herramientas, **({"tool_choice": tool_choice} if tool_choice else {}))
    return _etiquetar(llm, agente, modelo)


def crear_llm_estructurado(modelo: str, agente: str, esquema: type[BaseModel],
                           *, max_tokens: int = 500, razonamiento: Esfuerzo = "none") -> Runnable:
    """Salida estructurada con JSON Schema (`include_raw=True`: devuelve raw, parsed y parsing_error)."""
    llm = _chat(modelo, max_tokens, razonamiento).with_structured_output(
        esquema, method="json_schema", include_raw=True)
    return _etiquetar(llm, agente, modelo)


async def invocar_estructurado[T: BaseModel](llm: Runnable, mensajes: list[BaseMessage],
                                            config: RunnableConfig | None = None) -> tuple[T | None, str]:
    """Invoca un modelo estructurado sin lanzar por formato: devuelve (objeto | None, error).

    Un JSON inválido o una respuesta truncada por longitud se reportan como error
    para que el nodo degrade con elegancia; los errores de red sí se propagan
    (los reintenta la `RetryPolicy` del nodo).
    """
    try:
        r = await llm.ainvoke(mensajes, config)
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as e:
        return None, type(e).__name__
    return r["parsed"], "" if r["parsed"] is not None else repr(r["parsing_error"])[:120]


def texto(mensaje: BaseMessage) -> str:
    """Contenido de un mensaje como texto (algunos proveedores devuelven bloques)."""
    return mensaje.text if hasattr(mensaje, "text") and isinstance(mensaje.text, str) else str(mensaje.content)


def costo_estimado(modelo: str, t_in: int, t_out: int) -> float:
    p_in, p_out = PRECIOS.get(modelo, (0.0, 0.0))
    return (t_in * p_in + t_out * p_out) / 1e6


@dataclass(slots=True)
class Llamada:
    agente: str
    modelo: str
    inicio: float
    fin: float
    tokens_entrada: int
    tokens_salida: int
    tokens_cache: int
    costo_usd: float
    tool_calls: int
    error: str | None = None

    @property
    def latencia_s(self) -> float:
        return self.fin - self.inicio


ROLES_COORDINACION = frozenset({"router", "planificador", "critico"})


class Trazador(AsyncCallbackHandler):
    """Registra cada llamada a un modelo de una tarea y aplica el tope de gasto."""

    raise_error = True  # deja pasar PresupuestoExcedido hasta el grafo

    def __init__(self, tope_usd: float = float("inf")) -> None:
        self.tope_usd = tope_usd
        self.llamadas: list[Llamada] = []
        self._abiertas: dict[UUID, tuple[float, dict[str, Any]]] = {}

    @property
    def costo_usd(self) -> float:
        return sum(ll.costo_usd for ll in self.llamadas)

    async def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kw) -> None:
        if self.costo_usd > self.tope_usd:
            raise PresupuestoExcedido(f"Tarea por encima del tope de USD {self.tope_usd}")
        self._abiertas[run_id] = (time.perf_counter(), metadata or {})

    async def on_llm_end(self, response, *, run_id: UUID, **kw) -> None:
        t0, md = self._abiertas.pop(run_id, (time.perf_counter(), {}))
        msg = response.generations[0][0].message
        uso = getattr(msg, "usage_metadata", None) or {}
        t_in, t_out = uso.get("input_tokens", 0), uso.get("output_tokens", 0)
        cache = (uso.get("input_token_details") or {}).get("cache_read", 0) or 0
        crudo = (msg.response_metadata or {}).get("token_usage") or (response.llm_output or {}).get("token_usage") or {}
        modelo = md.get("modelo", "")
        costo = crudo.get("cost")
        if costo is None:
            costo = costo_estimado(modelo, t_in, t_out)
        self.llamadas.append(Llamada(md.get("agente", "?"), modelo, t0, time.perf_counter(), t_in, t_out,
                                     cache, float(costo), len(getattr(msg, "tool_calls", None) or [])))

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
        t0, md = self._abiertas.pop(run_id, (time.perf_counter(), {}))
        self.llamadas.append(Llamada(md.get("agente", "?"), md.get("modelo", ""), t0, time.perf_counter(),
                                     0, 0, 0, 0.0, 0, repr(error)[:200]))

    # --- Métricas ---------------------------------------------------------
    def por_agente(self) -> dict[str, dict[str, float]]:
        tabla: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for ll in self.llamadas:
            fila = tabla[ll.agente]
            fila["llamadas"] += 1
            fila["tokens_entrada"] += ll.tokens_entrada
            fila["tokens_salida"] += ll.tokens_salida
            fila["costo_usd"] += ll.costo_usd
            fila["latencia_s"] += ll.latencia_s
        return {k: dict(v) for k, v in tabla.items()}

    def pct_coordinacion(self) -> float:
        """Porcentaje del costo que se va en coordinar (router, planificador, crítico)."""
        total = self.costo_usd
        coord = sum(ll.costo_usd for ll in self.llamadas if ll.agente in ROLES_COORDINACION)
        return 100 * coord / total if total else 0.0

    def filas(self) -> list[dict[str, Any]]:
        return [asdict(ll) | {"latencia_s": ll.latencia_s} for ll in self.llamadas]
