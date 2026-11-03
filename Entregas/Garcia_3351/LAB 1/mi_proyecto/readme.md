# Ollama + Python Virtual Environment

Guía rápida para instalar **Ollama** y crear un entorno virtual de Python en Linux.

## 1. Instalar Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verificar:

```bash
ollama --version
```

Ejecutar un modelo:

```bash
ollama run qwen3:1.7b
```

Para ver los modelos instalados:

```bash
ollama list
```

---

## 2. Crear un entorno Python

Crear una carpeta de proyecto:

```bash
mkdir mi_proyecto
cd mi_proyecto
```

Crear el entorno:

```bash
python3 -m venv .venv
```

Activarlo:

```bash
source .venv/bin/activate
```

Cuando esté activo aparecerá:

```text
(.venv)
```

Python `venv` permite mantener las dependencias del proyecto aisladas del Python del sistema.

---

## 3. Instalar paquetes

Con el entorno activo:

```bash
pip install requests numpy
```

O instalar las dependencias de un proyecto:

```bash
pip install -r requirements.txt
```

---

## 4. Salir del entorno

```bash
deactivate
```

---

## Uso diario

Cada vez que vuelvas al proyecto:

```bash
cd mi_proyecto
source .venv/bin/activate
```

Ejecutar Ollama:

```bash
ollama run qwen3:1.7b
```

Salir del entorno:

```bash
deactivate
```

## Resumen

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Crear entorno Python
python3 -m venv .venv

# Activarlo
source .venv/bin/activate

# Instalar paquetes
pip install -r requirements.txt

# Ejecutar Ollama
ollama run qwen3:1.7b

# Salir
deactivate
```


