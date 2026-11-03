# Asistente del reglamento estudiantil de Icesi: informe

## Objetivo

Construir con LangGraph un sistema multiagente que responda dudas reales de estudiantes de pregrado de la Universidad Icesi a partir de documentos oficiales, citando el artículo que respalda cada regla. El sistema aplica los patrones de diseño vistos en clase y mide calidad, costo, latencia, robustez y costo de coordinación.

## Fuentes

- **Resolución de Rectoría 1.690 del 24 de marzo de 2026:** contiene el texto completo y vigente del *Libro de Derechos, Deberes y Normas de los Estudiantes de Pregrado*, con 110 artículos en 24 capítulos.
- **Guía oficial del servicio de cancelación de semestre en Banner.**

Ambas son públicas y se descargaron de icesi.edu.co. Un script convierte los PDF en 114 fragmentos (uno por artículo, más 4 secciones de la guía), con un total de 12.943 palabras. En la conversión se hicieron dos ajustes: unir las palabras que el PDF dejó partidas y reescribir como texto el cuadro de causales de retiro del Art. 68.

## Arquitectura

El sistema es un grafo de estados que combina cuatro patrones:

| Patrón | Función |
|---|---|
| Router | Reescribe la duda con el historial y decide la ruta: simple (un artículo o tema) o compleja (varios capítulos o un caso particular). |
| Orquestador–trabajadores | Un planificador divide la duda en temas (plazos, requisitos, consecuencias, costos), y cada tema lo resuelve un investigador con herramientas. |
| Paralelización | Los investigadores trabajan al mismo tiempo (`Send`) y sus notas se unen en el estado. |
| Evaluador–optimizador | Un crítico revisa el borrador y lo devuelve al redactor hasta aprobarlo, con un máximo de 2 rondas. |

```
duda -> router -> simple:   agente único ----------------------------------> verificador -> respuesta
               -> compleja: planificador -> investigadores (paralelo) -> redactor -> verificador -> crítico
                                                                            ^                         |
                                                                            +-------- corrige --------+
```

Los agentes trabajan con tres herramientas:
- **Buscador BM25:** usa raíces del español.
- **Lector de artículos completos:** sirve para seguir referencias como «según el Artículo 67».
- **Calculadora segura.**

Otros dos componentes completan el flujo:
- **Verificador de citas.** No usa LLM y comprueba que cada `[Art. N]` exista y se haya consultado, y que cada cifra aparezca en el artículo citado. Una cita inválida nunca se aprueba.
- **Revisión humana opcional.** Usa `interrupt` y permite que una persona apruebe o corrija la respuesta antes de entregarla.

## Decisiones de diseño

| Decisión | Elección |
|---|---|
| Control del flujo | El grafo fija las etapas; el LLM decide la ruta y las subtareas. |
| Comunicación | Mensajes estructurados validados con Pydantic. Si el formato es inválido, el sistema sigue por una opción segura y lo registra. |
| Memoria | El historial de la conversación se guarda por hilo, pero cada investigador ve solo su subtarea. |
| Modelo por rol | DeepSeek V4.1 Flash para trabajar, GPT-6 Luna como crítico (otra familia de modelo) y Claude Haiku 5.5 como juez. |
| Parada | Límite de pasos por agente, límite de rondas del crítico y tope de USD 0,05 por pregunta. |

## Evaluación

El dataset tiene 14 dudas reales de estudiantes. Cada una trae puntos clave y artículos esperados, tomados del texto vigente. Una de ellas, el valor de la matrícula, no la responde el reglamento, para comprobar que el sistema no invente.

Métricas usadas:
- **Calidad:** juez LLM con cobertura 60 %, fidelidad 25 % y claridad 15 %.
- **Recall de artículos:** fracción de los artículos esperados que el sistema llegó a consultar.

Se hizo una corrida, con un costo total de USD 0,136. El detalle está en `resultados/`.

| Configuración | Calidad | Cobertura | Recall de artículos | USD por duda | Latencia p50 / p95 | Éxito |
|---|---|---|---|---|---|---|
| A. Agente único | 0,832 | 0,91 | 0,84 | 0,00065 | 2,5 s / 8,2 s | 100 % |
| B. Multiagente | 0,867 | 0,97 | 0,96 | 0,00458 | 12,4 s / 28,8 s | 100 % |
| C. Router | 0,803 | 0,86 | 0,78 | 0,00145 | 4,2 s / 17,3 s | 93 % |
| D. Multiagente optimizado | **0,878** | **0,98** | **0,98** | 0,00203 | 7,5 s / 11,6 s | 100 % |

La configuración D usa 2 subtareas, 1 ronda del crítico y notas más cortas. La frontera de Pareto la forman A, la más barata, y D, la de mejor calidad.

## Conclusiones

1. **Con un corpus real y extenso, el multiagente sí aporta.** D encuentra el 98 % de los artículos necesarios, frente al 84 % del agente único. La diferencia se concentra en las dudas que cruzan capítulos: consecuencias de la prueba académica (0,90 frente a 0,69), causales de retiro (0,90 frente a 0,69) y doble programa (0,84 frente a 0,48). En el caso de prueba de la clase, que tiene 11 documentos cortos, había pasado lo contrario.
2. **La optimización paga.** D cuesta menos de la mitad que B y es más rápido, sin perder calidad. Dos investigadores enfocados y una sola ronda del crítico bastan.
3. **El router ahorra, pero en esta corrida no ganó en calidad.** Envió 9 de las 14 dudas por la ruta simple y cuesta un tercio de lo que cuesta B. Su promedio bajó por una tarea fallida; sin ella su calidad es 0,864, cercana a B, por la mitad del costo de D.
4. **Coordinar es barato frente a investigar.** Entre 12 y 15 % del costo se va en el router, el planificador y el crítico. El gasto grande son los tokens de lectura de los investigadores.

## Limitaciones

- **Hubo una sola repetición**, así que diferencias de pocas centésimas no son concluyentes.
- **La evaluación encontró un error de robustez:** una tarea del router falló porque el crítico devolvió un JSON truncado que lanzó una excepción. Se corrigió después de la corrida, con su prueba de regresión, y ahora el sistema entrega el borrador y registra el incidente. La corrida no se repitió.
- **El verificador no comprueba si el artículo citado es pertinente.** Comprueba que la cita exista y que las cifras coincidan. En una prueba, el sistema citó el Art. 62, que es de cursos de verano, como regla general de devolución, y el crítico lo aprobó.
- **La búsqueda es léxica.** Una búsqueda semántica podría mejorar el recall del agente único.

## Uso

```bash
uv sync
uv run asistente-icesi                  # conversación
uv run asistente-icesi --trazas "duda"  # muestra el costo por agente
uv run asistente-icesi-evaluar          # repite el experimento (usa crédito real)
uv run pytest                           # 35 pruebas sin costo
```
