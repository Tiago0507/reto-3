"""Pruebas de las métricas agregadas de la evaluación (sin llamadas a modelos)."""

from asistente_icesi.evaluacion import frontera_pareto, resumir


def _fila(config, calidad, costo, latencia):
    return {"config": config, "calidad": calidad, "costo_usd": costo, "latencia_s": latencia, "exito": True,
            "llamadas": 3, "pct_coordinacion": 20.0, "pct_consultas_repetidas": 0.0,
            "recall_articulos": 1.0}


def test_resumen_y_frontera_de_pareto():
    filas = [_fila("A", 0.7, 0.001, 2), _fila("A", 0.8, 0.001, 4),
             _fila("B", 0.9, 0.003, 9), _fila("C", 0.6, 0.004, 9)]
    resumen = {r["config"]: r for r in resumir(filas)}
    assert resumen["A"]["calidad"] == 0.75 and resumen["A"]["p50 s"] == 3
    assert frontera_pareto(list(resumen.values())) == ["A", "B"]  # C es más cara y peor: dominada
