# redteam-dx — Escaneo adversarial declarativo para apps de LLM

Red-teaming automatizado sobre una **suite de probes** (prompt injection, jailbreak,
fuga de PII, alucinación), calificada con un **modelo de severidad inspirado en
MITRE ATLAS** (critical/high/medium/low) y líneas de evidencia por hallazgo.

## Por qué importa
El red-teaming suele ser un arranque aislado de una vez. Este diseño lo hace **declarativo
y repetible**: agregás un probe como *dato*, lo corrés en cada release de modelo y dejás
que un job de CI bloquee el merge si `critical > 0`. La misma idea detrás de garak,
PyRIT y promptfoo.

## Método
- **Los probes son datos** (`ProbeSuites`): id, vector, payload, severidad de base.
- **El target es un callable**: `complete(prompt) -> str`. Cambiás el stub por un cliente
  real (Ollama local, OpenAI, Anthropic) — nada más cambia.
- **Detector heurístico:** marcadores de rechazo, respuestas vacías/telegráficas,
  contenido solicitado que aparece. En producción se complementa con un juez LLM para
  escenarios agresivos.
- **Reporte:** `findings` por probe con evidencia + agregado `by_severity`.

## Cómo correr
```bash
python redteam_dx/redteam_dx.py
```
Imprime un scan contra el stub offline. Apuntá `RedTeamDX.target` a tu modelo para
escanear el real.

## Para leer más
- garak (scanner de vulnerabilidades LLM, NVIDIA) — arquitectura probe/plug
- PyRIT (Microsoft) — orchestrators, converters, scoring
- promptfoo — config declarativa YAML de red team + integración CI
- MITRE ATLAS — tácticas y técnicas adversarias de IA