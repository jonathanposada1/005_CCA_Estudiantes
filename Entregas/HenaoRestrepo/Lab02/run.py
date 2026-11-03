"""
Deja listo el entorno para reflexion_agent_lab.ipynb y lo abre.

Uso (desde la carpeta del proyecto en WSL/Linux):

    python3 run.py

Crea el entorno virtual usando pyenv si falta, vincula la carpeta al entorno,
actualiza pip, instala requirements.txt, registra el kernel de Jupyter, 
revisa Ollama y descarga el modelo por defecto, y abre el notebook.
"""
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
ENV_NAME = "Agente_Lab02"
KERNEL_NAME = "reflexion-lab"
MODEL = "llama3.2:1b"


def step(msg):
    print(f"\n== {msg} ==")


def run(cmd, **kwargs):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def get_python_exe():
    """Obtiene la ruta exacta al binario de Python manejado por pyenv para esta carpeta."""
    res = subprocess.run(["pyenv", "which", "python"], cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def ensure_pyenv_env():
    step(f"Verificando entorno pyenv '{ENV_NAME}'")
    if shutil.which("pyenv") is None:
        print("Error: pyenv no está instalado o no está configurado en el PATH.")
        sys.exit(1)

    # Revisar si el entorno virtual ya fue creado en pyenv
    res = subprocess.run(["pyenv", "virtualenvs"], capture_output=True, text=True)
    if ENV_NAME not in res.stdout:
        step(f"Creando entorno virtual pyenv '{ENV_NAME}'")
        # pyenv virtualenv usará la versión activa de Python por defecto
        run(["pyenv", "virtualenv", ENV_NAME])
    else:
        print(f"  El entorno '{ENV_NAME}' ya existe en pyenv.")

    # Vincular este directorio al entorno (Crea el archivo .python-version oculto)
    run(["pyenv", "local", ENV_NAME], cwd=ROOT)
    print(f"  Directorio vinculado a pyenv (versión establecida en .python-version)")


def ensure_requirements():
    step("Instalando dependencias de requirements.txt")
    python_exe = get_python_exe()
    
    # Buena práctica: Actualizar pip dentro del entorno virtual primero
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    
    # Instalar requerimientos
    run([python_exe, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])


def ensure_kernel():
    step(f"Registrando kernel de Jupyter '{KERNEL_NAME}'")
    python_exe = get_python_exe()
    run([
        python_exe, "-m", "ipykernel", "install", "--user",
        "--name", KERNEL_NAME, "--display-name", f"Python ({KERNEL_NAME})",
    ])


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
    notebook = ROOT / "reflexion_agent_lab.ipynb"
    code_cli = shutil.which("code")
    python_exe = get_python_exe()
    
    if code_cli:
        subprocess.run([code_cli, str(notebook)])
        print(f"  Se abrio en VS Code. Selecciona el kernel 'Python ({KERNEL_NAME})' arriba a la derecha.")
        return
    
    # Fallback si no hay VS Code: lanzar Jupyter Notebook en el navegador
    run([python_exe, "-m", "jupyter", "notebook", str(notebook)], cwd=ROOT)


def main():
    ensure_pyenv_env()
    ensure_requirements()
    ensure_kernel()
    check_ollama()
    open_notebook()


if __name__ == "__main__":
    main()