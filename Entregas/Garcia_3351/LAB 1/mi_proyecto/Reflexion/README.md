# Agente con Reflexion: explicar conceptos de fisica (local, con Ollama)

Agente que usa el patron de **reflexion** (generar -> criticar -> revisar) para explicar conceptos de fisica a un estudiante, autocorrigiendose antes de entregar la respuesta final. El LLM corre **100% en tu computador** con [Ollama](https://ollama.com) — no necesitas ninguna API key ni conexion a internet.


## Que incluye

- `reflection_agent.py` — las tres funciones del ciclo (`explicar`, `revisar`, `corregir`) y `run_reflection`, que las encadena hasta que el revisor aprueba o se agotan los intentos.
- `display_helpers.py` — impresion legible de la traza del ciclo en el notebook.
- `reflexion_agent_lab.ipynb` — el notebook del lab, con ejemplos ya listos para correr.

## Opcion rapida

Con Python 3.9+ y [Ollama](https://ollama.com) instalados, desde la carpeta del proyecto:

```bash
python3 run.py
```

Esto crea el entorno virtual, instala dependencias, registra el kernel de Jupyter, descarga el modelo por defecto si falta, y abre el notebook. Es idempotente: puedes correrlo de nuevo sin romper nada.

## Paso a paso (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name reflexion-lab --display-name "Python (reflexion-lab)"
ollama pull llama3.2:1b
```

Luego abre `reflexion_agent_lab.ipynb`, selecciona el kernel **"Python (reflexion-lab)"** y corre las celdas en orden.

## Que esperar

- El modelo `llama3.2:1b` es pequeno: a veces el explicador comete un error, y a veces el revisor no lo detecta (o rechaza una explicacion que si estaba bien). El notebook incluye una seccion que compara esto contra un modelo mas grande (`qwen3:1.7b`), cambiando solo el parametro `model` de `run_reflection(...)`.
- Cada corrida puede dar una traza distinta (los LLMs no son deterministicos) — a veces la primera version ya es aprobada y no hay ronda de correccion.


----------------
La idea central — **un LLM que critica y corrige su propia salida antes de entregarla** — se traslada a otros flujos de un curso o laboratorio:

- **Retroalimentacion de tareas**: el agente redacta un comentario sobre la respuesta de un estudiante, y un segundo paso revisa que la retroalimentacion sea justa y este bien fundamentada antes de mostrarsela.
- **Otro criterio de revision**: cambia `REVISOR_PROMPT` en `reflection_agent.py` para que el revisor chequee otra cosa (ej. vocabulario adecuado al nivel del curso, o que incluya una analogia).
- **Ejercicio**:  Comparar la explicacion sin reflexion contra la explicacion final con reflexion para el mismo concepto — una forma concreta de ver que agrega (y que no arregla) la autocritica en un LLM.
- **Combinado con herramientas**: se puede encadenar con el patron de `../Tools` — por ejemplo, un agente que ajusta una curva y *despues* reflexiona sobre si el modelo elegido tiene sentido fisico.
