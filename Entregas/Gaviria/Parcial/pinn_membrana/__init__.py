"""PINN para la membrana cuadrada vibrante (Ejercicio 3)."""

from .analisis import EvaluadorSolucion, MetricasError, tabla
from .arquitecturas import ArquitecturaTemporal, ModeloA, ModeloB
from .fisica import MembranaCuadrada
from .optimizacion import Adam, EntrenadorPINN, ResultadoEntrenamiento
from .perdida import PerdidaPINN

__all__ = [
    "Adam",
    "ArquitecturaTemporal",
    "EntrenadorPINN",
    "EvaluadorSolucion",
    "MembranaCuadrada",
    "MetricasError",
    "ModeloA",
    "ModeloB",
    "PerdidaPINN",
    "ResultadoEntrenamiento",
    "tabla",
]
