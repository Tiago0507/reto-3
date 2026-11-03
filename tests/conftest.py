"""Modelo de chat falso y determinista para probar el grafo sin red ni costo.

Reemplaza la fábrica `_chat`: identifica el rol por el prompt de sistema y
responde con reglas simples. Las llamadas a herramientas son reales (ToolNode).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from asistente_icesi import modelos
from asistente_icesi.esquemas import Enrutamiento, Evaluacion, Plan


@dataclass
class Escenario:
    ruta: str = "simple"
    router_invalido: bool = False
    rechazos_critico: int = 0  # cuántas veces rechaza antes de aprobar
    cita_invalida: bool = False
    llamadas: list[str] = field(default_factory=list)


def _texto(m) -> str:
    return m.content if isinstance(m.content, str) else str(m.content)


class ChatFalso(BaseChatModel):
    escenario: Any

    @property
    def _llm_type(self) -> str:
        return "falso"

    def bind_tools(self, tools, tool_choice=None, **kw):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice)

    def with_structured_output(self, schema, *, include_raw=False, **kw):
        def responder(mensajes):
            sistema = _texto(mensajes[0])
            usuario = _texto(mensajes[-1])
            e = self.escenario
            if schema is Enrutamiento:
                e.llamadas.append("router")
                if e.router_invalido:
                    return {"raw": AIMessage("{"), "parsed": None, "parsing_error": ValueError("JSON roto")}
                obj = Enrutamiento(pregunta=usuario.split("Última pregunta: ")[-1], complejidad=e.ruta, razon="test")
            elif schema is Plan:
                e.llamadas.append("planificador")
                obj = Plan(subtareas=["plazos de cancelación de materias", "devolución de dinero",
                                      "límite de cancelaciones"])
            elif schema is Evaluacion:
                e.llamadas.append("critico")
                aprobado = e.rechazos_critico <= 0
                e.rechazos_critico -= 1
                obj = Evaluacion(aprobado=aprobado, puntaje=5 if aprobado else 2,
                                 comentarios="" if aprobado else "Agrega una recomendación explícita.")
            else:
                raise AssertionError(sistema)
            return {"raw": AIMessage(obj.model_dump_json()), "parsed": obj, "parsing_error": None}

        return RunnableLambda(responder)

    def _generate(self, messages, stop=None, run_manager=None, tools=None, tool_choice=None, **kw) -> ChatResult:
        sistema = _texto(messages[0]) if isinstance(messages[0], SystemMessage) else ""
        resultados = [m for m in messages if isinstance(m, ToolMessage)]
        humano = next(_texto(m) for m in messages if isinstance(m, HumanMessage))
        if tools and tool_choice != "none" and not resultados:
            self.escenario.llamadas.append("trabajador")
            msg = AIMessage("", tool_calls=[{"name": "buscar_reglamento", "args": {"consulta": humano[-120:]},
                                             "id": f"c_{uuid.uuid4().hex[:6]}", "type": "tool_call"}])
        elif tools:
            self.escenario.llamadas.append("trabajador")
            docs = json.loads(resultados[0].content) if resultados[0].content.startswith("[") else []
            cita = "Art. 999" if self.escenario.cita_invalida else (docs[0]["id"] if docs else "Art. 1")
            primera = docs[0]["texto"].split(". ")[0].rstrip(".") if docs else "Sin datos"
            msg = AIMessage(f"{primera} [{cita}].")  # una frase, citada
        else:
            assert "redactor" in sistema
            self.escenario.llamadas.append("redactor")
            notas = humano.split("Notas:")[-1].split("Borrador anterior:")[0]
            msg = AIMessage(" ".join(line for line in notas.splitlines() if line and not line.startswith("###")))
        msg.usage_metadata = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def escenario(monkeypatch) -> Escenario:
    e = Escenario()
    monkeypatch.setattr(modelos, "_chat", lambda modelo, max_tokens, razonamiento: ChatFalso(escenario=e))
    return e
