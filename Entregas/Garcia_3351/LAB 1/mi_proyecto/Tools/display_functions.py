"""Funciones para mostrar la traza del agente de forma legible en el notebook."""
import json


def pretty_print_agent_run(messages: list) -> None:
    """Imprime la traza completa de una corrida del agente: peticion, llamadas a herramientas y respuesta final."""
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)

        if role == "user":
            print(f"🧑 Peticion:\n{content}\n")
        elif role == "tool":
            print(f"🔧 Resultado de `{name}`:\n{content}\n")
        elif role == "assistant":
            if tool_calls:
                for call in tool_calls:
                    fn = call["function"] if isinstance(call, dict) else call.function
                    fn_name = fn["name"] if isinstance(fn, dict) else fn.name
                    fn_args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                    print(f"🤖 Llama a herramienta: {fn_name}({json.dumps(fn_args, ensure_ascii=False)})\n")
            if content:
                print(f"✅ Respuesta final:\n{content}\n")
