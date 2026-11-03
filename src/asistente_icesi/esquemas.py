"""Contratos entre agentes: salidas estructuradas validadas con Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Enrutamiento(BaseModel):
    """Decisión del router."""

    pregunta: str = Field(description="La pregunta actual reescrita para entenderse sin el historial.")
    complejidad: Literal["simple", "compleja"] = Field(
        description='"simple": la responde un solo artículo o tema del reglamento. '
        '"compleja": combina varios temas o capítulos, o analiza el caso particular del estudiante.'
    )
    razon: str = Field(description="Justificación en una frase.")


class Plan(BaseModel):
    """Plan del orquestador."""

    subtareas: list[str] = Field(
        min_length=1,
        description="Subtareas de investigación independientes, cada una sobre un tema distinto del reglamento.",
    )


class Evaluacion(BaseModel):
    """Veredicto del crítico."""

    aprobado: bool
    puntaje: int = Field(ge=1, le=5, description="1 = inaceptable, 5 = excelente.")
    comentarios: str = Field(description="Qué corregir, concreto y accionable. Vacío si se aprueba.")
