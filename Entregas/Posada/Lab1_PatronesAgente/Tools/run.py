"""
Deja listo el entorno para physics_agent_lab.ipynb y lo abre.

Uso (desde la carpeta del proyecto, en cualquier maquina):

    python3 run.py

Crea el entorno virtual si falta, instala requirements.txt, registra el
kernel de Jupyter, genera los datasets de ejemplo si no existen, revisa
Ollama y descarga el modelo por defecto, y abre el notebook.
"""
import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
KERNEL_NAME = "physics-agent-lab"
MODEL = "llama3.2:1b"
DATA_FILES = ["resorte.csv", "decaimiento.csv", "pendulo.csv"]


def step(msg):
    print(f"\n== {msg} ==")


def run(cmd, **kwargs):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_venv():
    if VENV_PYTHON.exists():
        return
    step("Creando entorno virtual (.venv)")
    venv.create(VENV_DIR, with_pip=True)


def ensure_requirements():
    step("Instalando dependencias de requirements.txt")
    run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])


def ensure_kernel():
    step(f"Registrando kernel de Jupyter '{KERNEL_NAME}'")
    run([
        str(VENV_PYTHON), "-m", "ipykernel", "install", "--user",
        "--name", KERNEL_NAME, "--display-name", f"Python ({KERNEL_NAME})",
    ])


def ensure_datasets():
    if all((ROOT / "data" / f).exists() for f in DATA_FILES):
        return
    step("Generando datasets de ejemplo")
    run([str(VENV_PYTHON), str(ROOT / "data" / "make_datasets.py")])


def ollama_running():
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except Exception:
        return False


def check_ollama():
    step("Revisando Ollama")
    if shutil.which("ollama") is None:
        print("  Ollama no esta instalado. Instalalo desde https://ollama.com,")
        print("  abrelo y vuelve a correr 'python3 run.py'.")
        return

    if not ollama_running():
        print("  Ollama esta instalado pero no esta corriendo.")
        print("  Abre la app de Ollama (o corre 'ollama serve' en otra terminal)")
        print("  y vuelve a correr 'python3 run.py' para descargar el modelo.")
        return

    try:
        modelos = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return
    if MODEL in modelos:
        print(f"  Modelo {MODEL} ya esta descargado.")
        return
    step(f"Descargando modelo {MODEL} (puede tardar unos minutos)")
    run(["ollama", "pull", MODEL])


def open_notebook():
    step("Abriendo el notebook")
    notebook = ROOT / "physics_agent_lab.ipynb"
    code_cli = shutil.which("code")
    if code_cli:
        subprocess.run([code_cli, str(notebook)])
        print(f"  Se abrio en VS Code. Selecciona el kernel 'Python ({KERNEL_NAME})' arriba a la derecha.")
        return
    run([str(VENV_PYTHON), "-m", "jupyter", "notebook", str(notebook)], cwd=ROOT)


def main():
    ensure_venv()
    ensure_requirements()
    ensure_kernel()
    ensure_datasets()
    check_ollama()
    open_notebook()


if __name__ == "__main__":
    main()
