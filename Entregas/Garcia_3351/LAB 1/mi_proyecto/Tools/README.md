# Agente de IA para ajuste de datos experimentales de física (local, con Ollama)

Un agente de IA que usa **herramientas de Python** (`scipy.optimize.curve_fit`, entre otras) para ajustar modelos a datos experimentales de física, siguiendo instrucciones en lenguaje natural. El LLM corre **100% en tu computador** con [Ollama](https://ollama.com) — no necesitas ninguna API key ni conexión a internet.

Está inspirado en el lab de "email assistant" de un curso de agentes de IA (`M3_UGL_2.ipynb`), pero aplicado a un caso de física en vez de correos.

## Qué incluye

- `data/make_datasets.py` — genera 3 datasets sintéticos (con ruido) de experimentos clásicos:
  - `resorte.csv` — Ley de Hooke (elongación vs fuerza) → modelo **lineal**.
  - `decaimiento.csv` — descarga de un circuito RC (tiempo vs voltaje) → modelo **exponencial**.
  - `pendulo.csv` — péndulo simple (longitud vs período) → modelo de **potencia**.
- `physics_tools.py` — las herramientas que usa el agente: listar datasets, previsualizarlos, listar modelos, ajustar curvas y graficar.
- `agent.py` — el ciclo que conecta el LLM (vía Ollama) con esas herramientas.
- `display_functions.py` — impresión legible de la traza del agente en el notebook.
- `physics_agent_lab.ipynb` — el notebook del lab, con ejemplos ya listos para correr.

## Opcion rapida

Con Python 3.9+ y [Ollama](https://ollama.com) instalados, desde la carpeta del proyecto:

```bash
python3 run.py
```

Esto crea el entorno virtual, instala dependencias, registra el kernel de Jupyter, genera los datasets si faltan, descarga el modelo por defecto si falta, y abre el notebook. Es idempotente: puedes correrlo de nuevo sin romper nada. Si prefieres hacerlo paso a paso (o entender que hace cada cosa), sigue las secciones 1 a 4 de abajo.

## 1. Requisitos

- Python 3.9 o superior.
- [Ollama](https://ollama.com) instalado y corriendo en tu computador.
- Al menos un modelo descargado, por ejemplo:
  ```bash
  ollama pull llama3.2:1b
  ```

## 2. Preparar el entorno

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Registra ese entorno como un kernel de Jupyter (para poder seleccionarlo al abrir el notebook):

```bash
python -m ipykernel install --user --name physics-agent-lab --display-name "Python (physics-agent-lab)"
```

## 3. Generar los datos de ejemplo

```bash
python data/make_datasets.py
```

Esto crea `data/resorte.csv`, `data/decaimiento.csv` y `data/pendulo.csv`.

## 4. Correr el notebook

1. Verifica que Ollama esté corriendo (`ollama list` debería mostrar tus modelos).
2. Abre `physics_agent_lab.ipynb` (en Jupyter Lab, Jupyter Notebook, o VS Code).
3. Selecciona el kernel **"Python (physics-agent-lab)"**.
4. Corre las celdas en orden.

## 5. Qué esperar

- El modelo `llama3.2:1b` es pequeño y rápido, pero no siempre confiable: a veces elige mal la herramienta o se equivoca al explicar un resultado. El notebook incluye una sección (6.5) que muestra esto y cómo cambiar a un modelo más grande (por ejemplo `qwen3:1.7b` o `llama3.2:3b`) sin tocar el resto del código — solo el parámetro `model` de `run_agent(...)`.
- Cada corrida puede dar una respuesta ligeramente distinta (los LLMs no son determinísticos). Los **números del ajuste** (`fit_dataset`) sí son siempre los mismos, porque esos vienen de `scipy`, no del LLM.

## Cómo aplicarlo en tus clases de física

La idea central — **darle a un LLM un conjunto de funciones ("herramientas") y dejar que decida cuál usar según una instrucción en lenguaje natural** — se traslada directamente a otros flujos típicos de un laboratorio o curso de física computacional:

- **Otro experimento**: agrega una función a `data/make_datasets.py` (o carga un CSV real de tus estudiantes) y regístralo en el diccionario `DATASETS` de `physics_tools.py`. No hace falta tocar el agente.
- **Otro modelo de ajuste**: agrega una función y su entrada al diccionario `MODELS` en `physics_tools.py` (por ejemplo, un modelo sinusoidal para un oscilador forzado, o una gaussiana para un pico espectral).
- **Otras herramientas**: por ejemplo, calcular el χ² reducido, propagar incertidumbres, comparar dos modelos con un criterio como AIC/BIC, o exportar resultados a una tabla — cada una es solo una función de Python más en `physics_tools.py`.
- **Modelos más grandes**: si tienes GPU o más RAM, un modelo Ollama más grande (`llama3.1:8b`, `qwen2.5:7b`, etc.) va a seguir instrucciones y elegir herramientas de forma mucho más confiable que `llama3.2:1b` — útil si quieres usar esto en una demo en vivo frente a estudiantes.
- **Posibles variantes**: pedirles que agreguen una herramienta nueva (ej. "ajuste con pesos", "detección de outliers") y que verifiquen si el agente la usa correctamente — es una forma concreta de que entiendan qué es y qué no es "razonamiento" en un LLM.
