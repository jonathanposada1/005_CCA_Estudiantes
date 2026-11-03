"""Evaluación de una solución entrenada: errores y energía."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .arquitecturas import ArquitecturaTemporal
from .fisica import MembranaCuadrada


@dataclass(frozen=True)
class MetricasError:
    l2_relativo: float
    linf: float
    rmse: float
    l2_campo: float
    residuo_rms: float

    def como_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass
class EvaluadorSolucion:
    modelo: ArquitecturaTemporal
    theta: np.ndarray
    membrana: MembranaCuadrada

    # ------------------------------------------------------------------ #
    def amplitud(self, t):
        return self.modelo.derivadas(t, self.theta)

    def residuo(self, t):
        a, _, d2a = self.amplitud(t)
        return d2a + self.membrana.kappa2 * a

    def campo(self, x, y, t):
        return self.membrana.campo(x, y, self.modelo(t, self.theta))

    def error_amplitud(self, t):
        return self.modelo(t, self.theta) - self.membrana.a_exacta(t)

    def error_campo(self, x, y, t):
        """U_theta - U = phi(x, y) [a_theta(t) - a(t)].

        Se evalúa en forma factorizada: restar los dos campos punto a punto
        añadiría un error de redondeo que no es parte del error del modelo.
        """
        return self.membrana.modo_espacial(x, y) * self.error_amplitud(t)

    # ------------------------------------------------------------------ #
    def metricas(self, t) -> MetricasError:
        """Errores sobre la malla temporal ``t`` (uniforme).

        ``l2_campo`` es la norma L2 del error de U en [0,1]^2 x [t_0, t_f]:
        como el error es phi(x, y) e(t) y int phi^2 = 1/4, vale ||e||_{L2} / 2.
        """
        t = np.asarray(t, dtype=float)
        e = self.error_amplitud(t)
        a_ex = self.membrana.a_exacta(t)
        norma_e = np.sqrt(trapezoid(e**2, t))
        return MetricasError(
            l2_relativo=float(norma_e / np.sqrt(trapezoid(a_ex**2, t))),
            linf=float(np.max(np.abs(e))),
            rmse=float(np.sqrt(np.mean(e**2))),
            l2_campo=float(0.5 * norma_e),
            residuo_rms=float(np.sqrt(np.mean(self.residuo(t) ** 2))),
        )

    # ------------------------------------------------------------------ #
    def energia(self, t):
        a, da, _ = self.amplitud(t)
        return self.membrana.energia_reducida(a, da)

    def energia_cuadratura(self, t, n_malla: int = 101):
        a, da, _ = self.amplitud(t)
        return self.membrana.energia_cuadratura(a, da, n_malla)

    def potencia(self, t):
        """dE/dt = a'(t) r(t) / 4: la energía sólo cambia si hay residuo."""
        _, da, _ = self.amplitud(t)
        return 0.25 * da * self.residuo(t)

    def deriva_energia(self, t) -> float:
        """max_t |E(t) - E_0| / E_0, con E_0 la energía exacta."""
        E0 = self.membrana.energia_inicial
        return float(np.max(np.abs(self.energia(t) - E0)) / E0)


def tabla(filas: dict[str, dict[str, float]], formato: str = "{:>12.3e}") -> str:
    """Tabla de texto con una fila por modelo y una columna por métrica."""
    columnas = list(next(iter(filas.values())))
    ancho = max(len(n) for n in filas) + 2
    encabezado = " " * ancho + "".join(f"{c:>14}" for c in columnas)
    lineas = [encabezado, "-" * len(encabezado)]
    for nombre, valores in filas.items():
        celdas = "".join("  " + formato.format(valores[c]) for c in columnas)
        lineas.append(f"{nombre:<{ancho}}{celdas}")
    return "\n".join(lineas)
