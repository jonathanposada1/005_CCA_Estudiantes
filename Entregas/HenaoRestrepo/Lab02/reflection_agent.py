"""
Ciclo de reflexión con Ollama local: generar -> criticar -> revisar.

A diferencia del agente de tool-calling (que decide que función de Python llamar), este agente no usa herramientas: el mismo LLM juega dos roles distintos -- "explicador" y "revisor" -- y el resultado se corrige a si mismo hasta que el revisor lo aprueba o se agotan los intentos.
"""
import ollama

EXPLICADOR_PROMPT = """\
Eres un profesor de física explicando un concepto a un estudiante de pregrado. 
Responde en español, en un párrafo corto y claro. 
No inventes formulas ni resultados: si no estas seguro de algo, dilo explicitamente.
"""

REVISOR_PROMPT = """\
Eres un revisor experto en física. 
Te dan la explicación que un profesor le dio a un estudiante sobre un concepto. 
Busca errores conceptuales, imprecisiones o falta de claridad.
No seas condescendiente rechaza si se requiere un cambio, se exigente.

Responde EXACTAMENTE en este formato (dos lineas):
<APROBADA> / <RECHAZADA>

COMENTARIOS: <que esta mal o que falta, en una o dos frases. 
Si no hay nada que corregir, escribe "Ninguno">
"""

def explicar(concepto: str, model: str) -> str:
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": EXPLICADOR_PROMPT},
        {"role": "user", "content": f"Explica: {concepto}"},
    ])
    return response["message"]["content"]

def revisar(concepto: str, explicacion: str, model: str) -> dict:
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": REVISOR_PROMPT},
        {"role": "user", "content": f"Concepto: {concepto}\n\nExplicación del profesor:\n{explicacion}"},
    ])
    texto = response["message"]["content"]
    texto_lower = texto.lower()
    aprobada = "aprobada" in texto_lower  and not "rechazada" in texto_lower
    return {"aprobada": aprobada, "comentarios": texto}

def corregir(concepto: str, explicacion: str, comentarios: str, model: str) -> str:
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": EXPLICADOR_PROMPT},
        {"role": "user", "content": f"Explica: {concepto}"},
        {"role": "assistant", "content": explicacion},
        {"role": "user", "content": (
            "Un revisor encontró este problema en tu explicación:\n"
            f"{comentarios}.\nEscribe una version corregida completa "
            "(no solo el cambio)."
        )},
    ])
    return response['message']['content']

def run_reflection(concepto: str, model: str = 'llama3.2:1b', max_rounds: int = 3) -> list:
    """
    Corre el ciclo generar -> criticar -> revisar y devuelve la traza completa.
    --------------------------------------------------------------------------
    Cada elemento de la traza tiene la explicación de esa ronda y la revision que recibió.
    ('None' si aun no se ha revisado, solo pasa con la ultima si se agotan las rondas).
    """
    explicacion = explicar(concepto, model)
    trace = [{'ronda': 1, 'explicacion': explicacion, 'revision': None}]
    
    for ronda in range(1, max_rounds+1):
        revision = revisar(concepto, explicacion, model)
        trace[-1]['revision'] = revision
        if revision['aprobada']:
            break
        # Si no aprobó y aún quedan rondas, se genera la corrección
        if ronda < max_rounds:
            explicacion = corregir(concepto, explicacion, revision['comentarios'], model)
            trace.append({'ronda': ronda + 1, 'explicacion': explicacion, 'revision': None})  
    return trace