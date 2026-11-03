"""Políticas de reintento de los nodos que llaman a modelos."""

from __future__ import annotations

from langgraph.types import RetryPolicy
from openai import ContentFilterFinishReasonError, LengthFinishReasonError

from .modelos import PresupuestoExcedido


def _reintentable(error: Exception) -> bool:
    """Reintenta fallas transitorias (red, límites de tasa, 5xx); nunca el tope de gasto ni errores de programación."""
    if isinstance(error, (PresupuestoExcedido, LengthFinishReasonError, ContentFilterFinishReasonError,
                          ValueError, TypeError, KeyError, AttributeError)):
        return False
    status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status_code", None)
    return status is None or status == 429 or status >= 500


REINTENTO_LLM = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0, retry_on=_reintentable)
