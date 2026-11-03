"""
Ciclo de tool-calling con Ollama local.

A diferencia de aisuite (que en el lab de email ejecutaba las tools
automaticamente via el parametro `max_turns`), el cliente `ollama` solo
genera la llamada a la herramienta: ejecutarla y devolverle el resultado
al modelo es responsabilidad nuestra. Este modulo implementa ese ciclo
explicitamente.
"""
import json

import ollama

SYSTEM_PROMPT = """\
Eres un asistente de fisica experimental especializado en ajuste de datos.
Puedes listar datasets y modelos disponibles, previsualizar datos, ajustar
un modelo a un dataset y generar graficas.
Usa las herramientas disponibles para responder. No inventes numeros: si
necesitas un valor, llama a la herramienta correspondiente.
Cuando termines, responde en espanol con un resumen claro y los numeros
obtenidos (incluye la incertidumbre si la calculaste).
"""


def run_agent(request: str, tools: list, model: str = "llama3.2:1b", max_turns: int = 5) -> list:
    """Ejecuta el ciclo agente-herramienta y devuelve la lista completa de mensajes (la traza)."""
    tools_by_name = {t.__name__: t for t in tools}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]

    for _ in range(max_turns):
        response = ollama.chat(model=model, messages=messages, tools=tools)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            fn = tools_by_name.get(name)
            if fn is None:
                result = json.dumps({"error": f"Herramienta '{name}' no disponible"}, ensure_ascii=False)
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = json.dumps({"error": str(exc)}, ensure_ascii=False)

            messages.append({"role": "tool", "content": str(result), "name": name})

    return messages
