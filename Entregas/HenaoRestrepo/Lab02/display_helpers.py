from typing import List, Dict, Any
from IPython.display import display, Markdown

def dMarkdown(text):
  # Displays the input string as Markdown.
  return display(Markdown(text))

def show_trace(concepto: str, trace: List[Dict[str, Any]]) -> None:
    """
    Impresión legible de la traza del ciclo de reflexión en el notebook.
    """
    # Concatenamos las líneas para renderizar un único bloque continuo en Jupyter
    md_lines = [f"## Concepto: {concepto}"]
    
    for paso in trace:
        md_lines.append(f"### Ronda {paso.get('ronda', 'N/A')}")
        md_lines.append(f"{paso.get('explicacion', '')}\n")
        
        revision = paso.get('revision')
        if revision:
            estado = 'Aprobada' if revision.get('aprobada') else "Rechazada, se corrige en la siguiente ronda"
            md_lines.append(f"**Revisión del crítico -- {estado}**<br>")
            md_lines.append(f"{revision.get('comentarios', '')}\n")
        # Separador visual sutil entre rondas
        md_lines.append("---") 
    # Unimos y mostramos todo de una sola vez
    dMarkdown("\n".join(md_lines))
