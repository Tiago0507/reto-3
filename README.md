# Asistente del reglamento estudiantil de Icesi (LangGraph)

Sistema multiagente que responde dudas de estudiantes de pregrado de la Universidad Icesi a partir de
documentos oficiales, citando el artículo exacto (`[Art. 53]`) de cada regla, plazo o cifra.

## Fuentes (reales, públicas)

| Documento | Uso |
|---|---|
| [Resolución de Rectoría 1.690 del 24 de marzo de 2026](https://www.icesi.edu.co/wp-content/uploads/2024/08/Reso-1690-Modifica-LDDN-reglamento-pregrado.pdf): texto completo y vigente del *Libro de Derechos, Deberes y Normas de los Estudiantes de Pregrado* | 110 artículos en 24 capítulos |
| [Guía del servicio de cancelación de semestre](https://www.icesi.edu.co/banner/images/nueva-imagen-institucional/estudiantes/Cancelación-de-semestre-estudiantes.pdf) (Banner) | 4 secciones: descripción, condiciones, pasos y contacto |

Los PDF están en `datos/fuentes/`. `uv run asistente-icesi-preparar` los convierte en `datos/corpus.json`:
- divide el texto en un fragmento por artículo;
- une las palabras que el PDF dejó partidas;
- reescribe como texto el cuadro del Art. 68.

## Arquitectura

```
START → enrutar ──simple───▶ responder_directo (ReAct) ──┐
           │                                             ▼
           └─compleja─▶ planificar ─Send×n─▶ investigar ─▶ redactar ─▶ verificar
                                     (en paralelo)          ▲             │ compleja
                                                            └─rechaza── criticar
                                                                          │ aprueba
                                      finalizar ◀── revision_humana ◀─────┘
                                          │          (interrupt opcional)
                                         END
```

| Patrón | Implementación |
|---|---|
| Router | `enrutar` reescribe la duda con el historial y elige la ruta: un artículo o tema va a `simple`; varios capítulos o un caso particular, a `compleja` |
| Orquestador–trabajadores | `planificar` divide la duda en temas (plazos, requisitos, consecuencias, costos); cada investigador es un subgrafo ReAct |
| Paralelización | `Send` por subtarea; el reductor `acumular` une las notas |
| Evaluador–optimizador | `criticar` (otra familia de modelo) devuelve el borrador a `redactar` hasta aprobarlo |
| Humano en el bucle | `interrupt()` con checkpointer (`--revisar`) |

Herramientas de los agentes:
- `buscar_reglamento`: búsqueda BM25 con raíces del español.
- `leer_articulo`: texto completo; sirve para seguir referencias del tipo «según el Artículo 67».
- `calcular`: calculadora segura.

El **verificador de citas** no usa LLM. Comprueba, frase por frase, que cada `[Art. N]` exista y se haya consultado, y que cada cifra aparezca en el artículo citado. Una cita inválida nunca se aprueba.

## Uso

```bash
uv sync
uv run asistente-icesi                                    # conversación con memoria
uv run asistente-icesi --trazas "¿Hasta qué semana puedo cancelar una materia?"
uv run asistente-icesi --revisar "¿Puedo estudiar dos carreras al mismo tiempo?"
uv run asistente-icesi --ruta simple|compleja "..."        # forzar la ruta
uv run asistente-icesi-evaluar                            # experimento A/B/C/D (gasta crédito real)
uv run pytest                                             # 35 pruebas con modelo falso, sin costo
uvx --from "langgraph-cli[inmem]" langgraph dev           # LangGraph Studio
```

La clave va en `.env` (ver `.env.example`). Una duda simple cuesta unos USD 0,0004; una compleja, unos USD 0,006.

## Evaluación

`evaluacion.py` contiene 14 dudas reales de estudiantes. Cada una tiene sus puntos clave y los artículos que la sustentan, tomados del texto vigente. El dataset incluye una pregunta que el reglamento no responde (el valor de la matrícula), para comprobar que el sistema no invente.

Métricas:
- **Calidad:** juez LLM de otra familia, con cobertura 60 %, fidelidad 25 % y claridad 15 %.
- **Recall de artículos:** qué parte de los artículos esperados llegó a consultar el sistema.
- **Costo, latencia (p50 y p95), tasa de éxito y porcentaje del costo en coordinación.**

Resultados de la primera corrida: el multiagente optimizado obtuvo la mejor calidad (0,878) y encontró el 98 % de los
artículos necesarios, frente al 84 % del agente único, por USD 0,002 por duda. El análisis completo está en
[INFORME.md](INFORME.md) y el detalle por duda, en `resultados/`.

## Estructura

```
datos/fuentes/          PDF oficiales
datos/corpus.json       corpus generado (114 fragmentos)
src/asistente_icesi/
  preparar.py           PDF → corpus.json
  corpus.py             carga y búsqueda BM25
  herramientas.py       buscar_reglamento, leer_articulo, calcular
  prompts.py            instrucciones por rol
  esquemas.py           contratos Pydantic (Enrutamiento, Plan, Evaluacion)
  estado.py             estado, entrada/salida y reductores
  agentes.py            subgrafo ReAct
  grafo.py              nodos y construcción del grafo
  verificacion.py       verificador de citas
  modelos.py            ChatOpenAI→OpenRouter, trazador de costo con tope
  configuracion.py      modelos por rol y Contexto (context_schema)
  cli.py, evaluacion.py
tests/                  modelo falso y pruebas
```
