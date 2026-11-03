"""Impresion legible de la traza del ciclo de reflexion en el notebook."""
from IPython.display import Markdown, display


def show_trace(concepto: str, trace: list) -> None:
    display(Markdown(f"## Concepto: {concepto}"))
    for paso in trace:
        display(Markdown(f"### Ronda {paso['ronda']}"))
        display(Markdown(paso["explicacion"]))

        revision = paso["revision"]
        if revision is None:
            continue
        estado = "Aprobada" if revision["aprobada"] else "Rechazada, se corrige en la siguiente ronda"
        display(Markdown(f"**Revision del critico -- {estado}**\n\n{revision['comentarios']}"))
