"""Pruebas de las piezas deterministas: corpus, búsqueda, herramientas, verificador y reductores."""

import pytest
from langchain_core.tools import ToolException

from asistente_icesi.configuracion import Contexto
from asistente_icesi.corpus import buscar, corpus, normalizar_id
from asistente_icesi.estado import REINICIAR, acumular
from asistente_icesi.herramientas import buscar_reglamento, calcular, leer_articulo
from asistente_icesi.verificacion import citas, verificar_citas


def _llamada(herramienta, **args):
    return herramienta.invoke({"name": herramienta.name, "args": args, "id": "1", "type": "tool_call"})


def test_corpus_tiene_los_110_articulos_y_la_guia():
    ids = [f["id"] for f in corpus()]
    assert ids[:110] == [f"Art. {n}" for n in range(1, 111)]
    assert ids[110:] == ["Guía 1", "Guía 2", "Guía 3", "Guía 4"]
    assert all(f["texto"] and f["capitulo"] for f in corpus())


@pytest.mark.parametrize(("consulta", "esperado"), [
    ("hasta qué semana puedo cancelar una materia", "Art. 53"),
    ("promedio para entrar en prueba académica", "Art. 67"),
    ("grado con honores cum laude", "Art. 83"),
    ("reclamo sobre una calificación", "Art. 77"),
    ("estudiar dos programas simultaneidad", "Art. 31"),
])
def test_busqueda_encuentra_el_articulo_relevante(consulta, esperado):
    assert esperado in [f["id"] for f in buscar(consulta)[:2]]


@pytest.mark.parametrize(("texto", "esperado"), [
    ("Art. 53", "Art. 53"), ("artículo 7", "Art. 7"), ("68", "Art. 68"), ("Guía 2", "Guía 2"), ("capítulo X", None)])
def test_normalizar_id(texto, esperado):
    assert normalizar_id(texto) == esperado


def test_herramientas_devuelven_ids_como_artifact():
    msg = _llamada(buscar_reglamento, consulta="cancelación de materias")
    assert "Art. 53" in msg.artifact and "duodécima" in msg.content
    msg = _llamada(leer_articulo, identificador="artículo 68")
    assert msg.artifact == ["Art. 68"] and "tercera repetición" in msg.content


def test_leer_articulo_inexistente_es_error_de_herramienta():
    with pytest.raises(ToolException):
        leer_articulo.func("Art. 500")


@pytest.mark.parametrize(("expr", "esperado"), [("(4.2*3 + 3.8*4) / 7", "3.97143"), ("0,8 * 64", "51.2")])
def test_calcular(expr, esperado):
    assert calcular.invoke(expr) == esperado


@pytest.mark.parametrize("expr", ["__import__('os')", "a + 1", "9 ** 9999", "1/0"])
def test_calcular_rechaza_expresiones_inseguras(expr):
    with pytest.raises(ToolException):
        calcular.func(expr)


def test_citas_acepta_variantes():
    assert citas("Ver [Art. 50, Art. 53] y [Artículo 54] y [Guía 2].") == ["Art. 50", "Art. 53", "Art. 54", "Guía 2"]


def test_verificador_acepta_cifras_respaldadas():
    v = verificar_citas("Entra en prueba académica con promedio inferior a 3,3 tras el segundo semestre y "
                        "a 3,4 en los siguientes [Art. 67]. Cancelar es posible hasta la semana 12 [Art. 53].",
                        {"Art. 67", "Art. 53"})
    assert v["fidelidad"] == 1.0 and not v["citas_invalidas"] and not v["cifras_sin_cita"]


def test_verificador_detecta_problemas():
    v = verificar_citas("El cum laude exige 4,1 [Art. 83]. Se pierde con 70 % de asistencia. Ver [Art. 999].",
                        {"Art. 83"})
    assert v["citas_invalidas"] == ["Art. 999"]
    assert v["cifras_sin_respaldo"] and v["cifras_sin_cita"]
    assert v["fidelidad"] == 0.0


def test_verificador_exige_articulo_consultado():
    assert verificar_citas("Plazo de 3 días hábiles [Art. 77].", consultados=set())["citas_invalidas"] == ["Art. 77"]


def test_reductor_acumula_y_reinicia():
    assert acumular(["a"], ["b"]) == ["a", "b"]
    assert acumular(["a"], REINICIAR) == []


def test_contexto_valida_limites():
    with pytest.raises(ValueError):
        Contexto(max_subtareas=9)
    with pytest.raises(ValueError):
        Contexto(tope_usd=0)
