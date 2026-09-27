# arena — Comparación head-to-head de modelos

Compara varios LLMs (distintos proveedores Y modelos locales) sobre un mismo set de
prompts, y reporta un ranking tipo arena con **intervalos de confianza vía bootstrap**.

## Por qué importa
"¿Cuál modelo es mejor?" es una pregunta *pareada*, no de un solo número. Un único F1
sobre un golden set puede esconder un cambio de 2 puestos en el ranking. Esta mini-herramienta
trae el método **Chatbot Arena / Bradley–Terry** a una evaluación controlada y reproducible:
vos elegís los prompts, el juez y el set de modelos.

## Método
- **Batallas:** emparejamientos aleatorios round-robin sobre tu set de prompts, juzgados
  por un árbitro (juez LLM o regla determinística).
- **Win rate:** los empates cuentan como media victoria.
- **Bradley–Terry:** ascenso de gradiente que maximiza la verosimilitud pareada → fuerza
  por modelo, recentrada a una escala tipo Elo.
- **Bootstrap CI:** remuestrear la lista de partidas con reemplazo 500×, recalcular
  fuerzas y reportar el percentil 2.5–97.5.

## Cómo correr
```bash
python arena/demo.py
```
El demo usa responders determinísticos offline, así funciona con cero dependencias y
imprime un ranking reproducible. Reemplazá `_mock_responder` por clientes reales
(OpenAI, Anthropic, Ollama/vLLM local) para correr contra tus modelos reales.

## Qué mirar
- CI ancho ⇒ ranking inestable ⇒ necesitás más batallas o mejores prompts.
- El sesgo del juez es el principal confundidor: medí el acuerdo del juez (kappa) antes
  de confiar en la tabla.

## Memoria técnica
- Metodología Chatbot Arena (Bradley–Terry + Elo)
- Intervalos de confianza bootstrap para modelos BT