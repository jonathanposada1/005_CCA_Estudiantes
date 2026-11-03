# Laboratorio 2 — Agente de Inteligencia Artificial

**Universidad de Antioquia**  
**Estudiante:** Santiago Ramírez Puentes  
**Laboratorio:** 2  
**Año:** 2026  

---

## Descripción

Este laboratorio presenta la implementación y despliegue de un **sistema de agentes de inteligencia artificial orientado a tareas de investigación y generación de reportes**.

La aplicación está desarrollada principalmente en **Python** y utiliza **FastAPI** para exponer los servicios del sistema y proporcionar una interfaz de interacción. El modelo de lenguaje se ejecuta localmente mediante **Ollama**, mientras que las herramientas de investigación permiten complementar las capacidades del agente con búsqueda de información externa.

El proyecto se ejecuta dentro de un entorno **Docker**, lo que permite encapsular las dependencias y facilitar su reproducción en diferentes equipos. Adicionalmente, se utiliza **PostgreSQL** como componente de persistencia durante la ejecución.

---

## Objetivo

El objetivo del laboratorio es implementar una arquitectura de agentes capaz de recibir una solicitud del usuario y coordinar diferentes componentes para:

- analizar la solicitud;
- planificar la tarea;
- realizar procesos de investigación;
- utilizar herramientas externas cuando sea necesario;
- interactuar con un modelo de lenguaje local;
- sintetizar la información recuperada;
- generar un reporte;
- y permitir consultar el progreso de la tarea.

El sistema busca mostrar cómo un agente de inteligencia artificial puede ir más allá de una interacción directa con un modelo de lenguaje, incorporando planificación, herramientas de investigación y diferentes etapas de procesamiento.

---

## Arquitectura general

El funcionamiento general del sistema puede representarse de la siguiente manera:

```text
                    ┌──────────────────────┐
                    │       Usuario        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Interfaz Web      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sistema de Agentes   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Planning Agent  │        │ Research Tools  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 │                          ▼
                 │                    ┌───────────┐
                 │                    │  Tavily   │
                 │                    │  arXiv    │
                 │                    └───────────┘
                 │
                 ▼
        ┌─────────────────┐
        │     Ollama      │
        │   Modelo LLM    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Writer / Editor │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Reporte final   │
        └─────────────────┘
```

Docker proporciona el entorno de ejecución de la aplicación y PostgreSQL se inicializa como parte de la infraestructura utilizada por el proyecto.

---

## Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| **Python** | Lenguaje principal del proyecto |
| **FastAPI** | API y servidor de la aplicación |
| **Uvicorn** | Servidor ASGI |
| **Docker** | Contenerización del sistema |
| **PostgreSQL** | Persistencia de información |
| **Ollama** | Ejecución local del modelo de lenguaje |
| **Qwen3** | Modelo de lenguaje utilizado en la ejecución local |
| **Tavily API** | Herramienta de búsqueda para investigación |
| **arXiv** | Fuente de literatura académica |
| **HTML / CSS** | Interfaz web |
| **GitHub** | Control de versiones y entrega del proyecto |

---

## Estructura del proyecto

El código fuente se encuentra en:

```text
Sesion03_agentes-ai-public-v2/
```

La estructura principal es:

```text
Sesion03_agentes-ai-public-v2/
│
├── docker/
│   └── entrypoint.sh
│
├── scripts/
│   └── test_ollama_connection.py
│
├── src/
│   ├── agents.py
│   ├── planning_agent.py
│   └── research_tools.py
│
├── static/
│   ├── arxiv_logo.png
│   ├── dl_logo.png
│   ├── how_interactions_are_performed.png
│   ├── tavily_logo.svg
│   └── wikipedia_logo.png
│
├── templates/
│   └── index.html
│
├── .env.example
├── .gitignore
├── Dockerfile
├── Informe_Docker_Windows.docx.pdf
├── main.py
└── requirements.txt
```

---

## Componentes principales

### `main.py`

Contiene la aplicación desarrollada con FastAPI. Desde este archivo se administra la interacción entre la interfaz web y el sistema encargado de procesar las solicitudes.

También expone los endpoints utilizados para iniciar y consultar las tareas del agente.

### `src/agents.py`

Contiene la lógica asociada con los diferentes agentes y la interacción con el modelo de lenguaje.

Los agentes permiten separar responsabilidades dentro del proceso de investigación, redacción y revisión del contenido.

### `src/planning_agent.py`

Implementa la lógica relacionada con la planificación de las tareas que debe realizar el sistema.

A partir de la solicitud del usuario, el sistema puede generar diferentes pasos de investigación antes de comenzar la ejecución.

### `src/research_tools.py`

Contiene herramientas utilizadas durante la etapa de investigación y recuperación de información.

Estas herramientas permiten complementar las capacidades del modelo de lenguaje mediante fuentes externas.

### `scripts/test_ollama_connection.py`

Permite realizar pruebas relacionadas con la comunicación entre la aplicación y Ollama.

### `templates/index.html`

Define la interfaz web principal desde la cual el usuario puede interactuar con la aplicación y observar el progreso de las diferentes etapas.

### `docker/entrypoint.sh`

Inicializa los servicios requeridos dentro del contenedor, configura PostgreSQL y posteriormente inicia el servidor de la aplicación.

### `Dockerfile`

Define el entorno Docker del proyecto, instala las dependencias necesarias y configura la ejecución de la aplicación.

---

## Flujo de agentes

Durante una ejecución, el sistema puede dividir una solicitud en diferentes etapas.

En la prueba realizada durante el laboratorio se observó un flujo compuesto por tareas como:

```text
Solicitud del usuario
        │
        ▼
Planning Agent
        │
        ▼
Generación del plan de investigación
        │
        ├───────────────┐
        ▼               ▼
Research Agent      Research Agent
   Tavily               arXiv
        │               │
        └───────┬───────┘
                ▼
          Writer Agent
                │
                ▼
          Editor Agent
                │
                ▼
          Writer Agent
                │
                ▼
           Reporte final
```

De esta manera, una sola consulta del usuario puede producir múltiples etapas internas. Esto diferencia el sistema de una interacción convencional en la que simplemente se envía una pregunta directamente al modelo de lenguaje.

---

## Requisitos

Para ejecutar el proyecto se requiere tener instalado:

- **Docker Desktop**
- **Ollama**
- un modelo compatible disponible localmente en Ollama;
- una clave válida para **Tavily API**;
- conexión a Internet para las funcionalidades que utilizan servicios externos.

Las dependencias de Python utilizadas por la aplicación se encuentran definidas en:

```text
requirements.txt
```

y son instaladas durante la construcción de la imagen Docker.

---

## Configuración de variables de entorno

El proyecto utiliza un archivo:

```text
.env
```

para almacenar la configuración local y las credenciales necesarias.

Por razones de seguridad, el archivo `.env` real **no se encuentra incluido en este repositorio**.

En su lugar se proporciona:

```text
.env.example
```

como plantilla de configuración.

Para ejecutar el proyecto se debe crear una copia de `.env.example`.

En Linux puede realizarse mediante:

```bash
cp .env.example .env
```

En Windows también puede realizarse manualmente copiando el archivo y cambiando su nombre a:

```text
.env
```

Posteriormente deben configurarse las variables correspondientes, especialmente la clave de Tavily, el modelo disponible localmente y la dirección utilizada para acceder a Ollama.

> **Importante:** `.env.example` constituye una plantilla de referencia. La configuración de `.env` puede variar dependiendo del sistema operativo, los modelos instalados y la forma en que Docker accede al servicio de Ollama.

Nunca se deben publicar claves reales de API en GitHub.

---

# Ejecución

## 1. Iniciar Docker Desktop

Abrir **Docker Desktop** y esperar hasta que el motor de Docker se encuentre disponible.

Puede comprobarse mediante:

```bash
docker version
```

---

## 2. Verificar Ollama

Comprobar que Ollama se encuentra instalado:

```bash
ollama --version
```

Consultar los modelos disponibles localmente:

```bash
ollama list
```

También puede comprobarse que el servidor de Ollama está respondiendo mediante:

```bash
curl http://localhost:11434/api/tags
```

En PowerShell, si `curl` utiliza `Invoke-WebRequest`, también puede utilizarse:

```powershell
curl.exe http://localhost:11434/api/tags
```

El modelo especificado en `.env` debe coincidir con uno de los modelos instalados localmente en Ollama.

En la ejecución realizada para este laboratorio se utilizó:

```text
qwen3:1.7b
```

Si este modelo no se encuentra instalado, puede descargarse mediante:

```bash
ollama pull qwen3:1.7b
```

---

## 3. Configuración utilizada en Windows

Durante la ejecución y validación del laboratorio en Windows se utilizó la siguiente configuración local:

```env
LLM_MODEL=openai:qwen3:1.7b
OLLAMA_API_URL=http://host.docker.internal:11434
OLLAMA_TIMEOUT=300
TAVILY_API_KEY=TU_API_KEY
```

La clave:

```text
TAVILY_API_KEY
```

debe reemplazarse localmente por una credencial válida del usuario.

### ¿Por qué aparece `openai` si se utiliza Ollama?

Aunque la variable:

```text
LLM_MODEL=openai:qwen3:1.7b
```

contiene el prefijo `openai`, el modelo utilizado durante esta configuración se ejecuta **localmente mediante Ollama**.

El proyecto utiliza una interfaz compatible con OpenAI para realizar la comunicación con el servidor local de Ollama.

Por tanto, en esta configuración:

```text
Aplicación
    │
    ▼
Interfaz compatible con OpenAI
    │
    ▼
Ollama
    │
    ▼
qwen3:1.7b
```

el modelo sigue ejecutándose localmente.

### Comunicación entre Docker y Ollama en Windows

Ollama se ejecuta en el sistema anfitrión, mientras que la aplicación se ejecuta dentro de Docker.

Por esta razón, dentro del contenedor:

```text
127.0.0.1
```

hace referencia al propio contenedor y no necesariamente al computador anfitrión.

En la configuración validada en Windows se utilizó:

```text
http://host.docker.internal:11434
```

permitiendo el siguiente flujo:

```text
Contenedor Docker
       │
       │ host.docker.internal:11434
       ▼
Windows
       │
       ▼
Ollama
       │
       ▼
qwen3:1.7b
```

---

## 4. Ubicarse en la carpeta del proyecto

Desde una terminal, ingresar a:

```text
Sesion03_agentes-ai-public-v2
```

Todos los siguientes comandos deben ejecutarse desde esta carpeta.

---

## 5. Construir la imagen Docker

Ejecutar:

```bash
docker build -t agentic-ai-v2 .
```

Este comando:

1. crea la imagen Docker;
2. instala las dependencias de Python;
3. copia el código de la aplicación;
4. configura el script de inicio;
5. y prepara el entorno necesario para ejecutar el sistema.

La imagen resultante recibe el nombre:

```text
agentic-ai-v2
```

---

## 6. Ejecutar el contenedor

En Windows se puede iniciar la aplicación mediante:

```bash
docker run --rm -it -p 8000:8000 --env-file .env --name fpsvc-v2 agentic-ai-v2
```

El parámetro:

```text
-p 8000:8000
```

establece la comunicación entre el puerto `8000` del computador anfitrión y el puerto `8000` del contenedor.

El parámetro:

```text
--env-file .env
```

carga las variables de entorno requeridas por la aplicación.

Durante el inicio se configura PostgreSQL y posteriormente se ejecuta FastAPI mediante Uvicorn.

Una ejecución correcta debe mostrar mensajes similares a:

```text
Starting Postgres cluster 17/main...
Postgres is ready
DATABASE_URL=postgresql://app:local@127.0.0.1:5432/appdb

INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Interfaz de usuario

Con el contenedor ejecutándose, la interfaz principal puede abrirse desde el navegador en:

```text
http://localhost:8000/
```

Desde esta interfaz el usuario puede introducir un tema de investigación y observar las diferentes etapas ejecutadas por el sistema.

---

## Documentación de la API

FastAPI genera automáticamente una interfaz para consultar y probar los endpoints disponibles.

Se encuentra en:

```text
http://localhost:8000/docs
```

Los principales endpoints de la aplicación son:

| Método | Endpoint | Función |
|---|---|---|
| `GET` | `/` | Carga la interfaz principal |
| `GET` | `/api` | Verificación de la API |
| `POST` | `/generate_report` | Inicia la generación de un reporte |
| `GET` | `/task_progress/{task_id}` | Consulta el progreso de una tarea |
| `GET` | `/task_status/{task_id}` | Consulta el estado de una tarea |

---

## Ejemplo de ejecución

Para comprobar el funcionamiento del agente se puede utilizar una solicitud como:

```text
Investiga brevemente qué es el entrelazamiento cuántico y genera un resumen corto.
```

A partir de la solicitud, el Planning Agent genera automáticamente un conjunto de pasos.

Durante la prueba realizada se observaron etapas correspondientes a:

1. planificación de la investigación;
2. búsqueda amplia mediante Tavily;
3. búsqueda de literatura académica mediante arXiv;
4. síntesis inicial mediante un Writer Agent;
5. revisión mediante un Editor Agent;
6. elaboración del reporte;
7. revisión y generación del resultado final.

Esto permite comprobar la coordinación entre el modelo de lenguaje y las diferentes herramientas disponibles.

---

## ¿Por qué la generación puede tardar?

La generación de un reporte puede tardar más que una conversación convencional con un modelo de lenguaje debido a que el sistema ejecuta múltiples operaciones.

Una consulta puede requerir:

```text
Planificación
     +
Búsquedas externas
     +
Procesamiento de resultados
     +
Múltiples llamadas al LLM
     +
Redacción
     +
Revisión
     +
Generación del reporte final
```

Además, el modelo se ejecuta localmente mediante Ollama, por lo que la velocidad depende de los recursos computacionales disponibles en el equipo.

Por esta razón, una tarea de investigación completa puede requerir más tiempo que una respuesta directa de un chatbot.

---

## Docker y PostgreSQL

Docker permite encapsular las dependencias de la aplicación y proporcionar un entorno reproducible.

Durante el inicio del contenedor, el script:

```text
docker/entrypoint.sh
```

se encarga de preparar PostgreSQL y posteriormente iniciar la aplicación mediante Uvicorn.

De manera simplificada:

```text
┌─────────────────────────────┐
│      Contenedor Docker      │
│                             │
│  ┌───────────────────────┐  │
│  │       FastAPI         │  │
│  └──────────┬────────────┘  │
│             │               │
│  ┌──────────▼────────────┐  │
│  │ Sistema de agentes    │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │      PostgreSQL       │  │
│  └───────────────────────┘  │
│                             │
└─────────────┬───────────────┘
              │
              ▼
       Ollama / Tavily
       / fuentes externas
```

---

## Compatibilidad entre Windows y Linux

Durante la ejecución del proyecto en Windows se presentó una diferencia entre los finales de línea utilizados por Windows (`CRLF`) y Linux (`LF`).

Esto podía producir el error:

```text
env: 'bash\r': No such file or directory
```

al intentar ejecutar:

```text
docker/entrypoint.sh
```

Para garantizar la compatibilidad, el `Dockerfile` normaliza los finales de línea antes de ejecutar el script:

```dockerfile
RUN sed -i 's/\r$//' /entrypoint.sh
```

Posteriormente se asignan los permisos de ejecución correspondientes.

Esta modificación permite construir la imagen desde Windows y ejecutar correctamente el script dentro del entorno Linux utilizado por Docker.

---

## Detener la aplicación

Si el contenedor se está ejecutando en primer plano, puede detenerse mediante:

```text
Ctrl + C
```

También puede detenerse desde otra terminal:

```bash
docker stop fpsvc-v2
```

El parámetro:

```text
--rm
```

utilizado al iniciar el contenedor hace que Docker elimine automáticamente el contenedor cuando finaliza su ejecución.

La imagen:

```text
agentic-ai-v2
```

permanece disponible y puede utilizarse nuevamente.

Por lo tanto, no es necesario ejecutar:

```bash
docker build -t agentic-ai-v2 .
```

cada vez que se inicia el proyecto, salvo que se hayan realizado cambios en el código, dependencias o configuración de construcción que requieran reconstruir la imagen.

---

## Seguridad

El proyecto utiliza `.gitignore` para evitar que información privada sea incorporada al repositorio.

En particular:

```gitignore
.env
```

impide versionar el archivo local que puede contener credenciales.

La distribución correcta es:

```text
.env           → configuración privada, no se publica
.env.example   → plantilla pública de configuración
```

Cada usuario debe proporcionar sus propias credenciales antes de ejecutar el proyecto.

En particular, una clave real de Tavily nunca debe almacenarse públicamente en GitHub.

---

## Reproducibilidad

Una de las principales ventajas de la implementación es la posibilidad de reproducir el entorno utilizando Docker.

El procedimiento general en un nuevo equipo es:

```text
1. Obtener el proyecto
        ↓
2. Instalar/iniciar Docker Desktop
        ↓
3. Instalar/iniciar Ollama
        ↓
4. Instalar un modelo compatible
        ↓
5. Crear .env a partir de .env.example
        ↓
6. Configurar el modelo y las credenciales
        ↓
7. Construir la imagen Docker
        ↓
8. Ejecutar el contenedor
        ↓
9. Abrir http://localhost:8000/
```

Esto reduce los problemas asociados con diferencias entre versiones de Python, dependencias y configuraciones locales.

---

## Validación del funcionamiento

Durante la ejecución del laboratorio se verificó el funcionamiento de los principales componentes del sistema.

Se comprobó:

- inicialización correcta del contenedor Docker;
- inicialización de PostgreSQL;
- ejecución de FastAPI mediante Uvicorn;
- acceso a la interfaz web;
- comunicación entre Docker y Ollama;
- utilización del modelo local `qwen3:1.7b`;
- generación automática del plan de investigación;
- ejecución de búsquedas mediante Tavily;
- consulta de literatura académica mediante arXiv;
- ejecución de etapas de escritura;
- ejecución de etapas de edición y revisión.

Esto permite validar el funcionamiento integrado del sistema y no únicamente el inicio de la API.

---

## Resultado del laboratorio

El resultado final corresponde a una aplicación de inteligencia artificial con una arquitectura modular que integra:

- agentes especializados;
- planificación automática de tareas;
- herramientas de investigación;
- recuperación de información externa;
- consulta de literatura académica;
- un modelo de lenguaje ejecutado localmente mediante Ollama;
- agentes de escritura y edición;
- una API REST desarrollada con FastAPI;
- una interfaz web;
- PostgreSQL;
- y un entorno reproducible mediante Docker.

La implementación permite observar cómo un sistema basado en agentes puede combinar un modelo de lenguaje con herramientas, planificación e infraestructura de software para desarrollar tareas más complejas que una interacción directa con un LLM.

En particular, el sistema puede recibir un tema de investigación, generar un plan, utilizar fuentes externas, procesar la información recuperada y coordinar diferentes agentes para producir un reporte final.

---

## Autor

**Santiago Ramírez Puentes**  
Universidad de Antioquia  
2026

