# eval-watch — Detección de drift + gates de calidad de 3 escenarios

Compara continuamente la distribución de una métrica (latencia, exactitud, costo)
contra un baseline congelado, levanta una **alerta de drift** vía PSI/KL y decide
sobre releases con un **gate de 3 escenarios** (pesimista / base / optimista).

## Por qué importa
Los umbrales de un solo número mienten. Un release puede pasar el "promedio de
latencia ok" mientras el p95 empeora bajo carga. El PSI captura el *cambio de
distribución* que tu media esconde; el gate de 3 escenarios fuerza a shippear solo
cuando también se cumple el caso pesimista — la disciplina que exigen los dominios
regulados.

## Método
- **PSI**: distancia de distribución discreta sobre scores agrupados en bins —
  0 = idéntico, zona de alerta convencional >0.1.
- **KL divergence**: `KL(actual || expected)`, sensible a desplazamientos de cola.
- **Bootstrap CI** sobre la métrica → pesim./base/optim. → gate
  `PASS | CONDITIONAL | BLOCK`.

## Cómo correr
```bash
python eval_watch/eval_watch.py
```
Demo offline: baseline vs. una distribución con drift; imprime PSI/KL + decisión del gate.

## Dónde encaja
Ubicado detrás de tu endpoint de modelo o runner de harness, logueá cada corrida al
store de baselines, alertá en PSI > 0.1 y bloqueá merges cuando el gate diga `BLOCK`.

## Memoria técnica
- PSI en monitoreo de riesgo crediticio (alerta estándar >0.1)
- KL divergence para drift en ML (enfoque de Evidently)