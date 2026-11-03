"""Física de la membrana cuadrada con bordes fijos.

Contiene la solución analítica del problema de valor inicial y las
expresiones de la energía mecánica, tanto la reducida (en términos de la
amplitud modal a(t)) como la obtenida por cuadratura numérica sobre la
membrana, que sirve como verificación independiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson


@dataclass(frozen=True)
class MembranaCuadrada:
    """Membrana de lado unitario excitada en el modo fundamental (1, 1).

    Con U = phi(x, y) a(t) y phi = sin(pi x) sin(pi y), la ecuación de ondas
    se reduce a  a'' + kappa^2 a = 0,  con kappa^2 = 2 pi^2.

    Parámetros
    ----------
    u0 : float
        Amplitud inicial, a(0).
    v0 : float
        Velocidad inicial de la amplitud, a'(0).
    """

    u0: float = 1.0
    v0: float = float(np.sqrt(2.0 * np.pi))
    kappa2: float = field(init=False, default=2.0 * np.pi**2)

    # ------------------------------------------------------------------ #
    # Constantes del modo
    # ------------------------------------------------------------------ #
    @property
    def omega(self) -> float:
        """Frecuencia angular del modo fundamental, sqrt(2) pi."""
        return float(np.sqrt(self.kappa2))

    @property
    def periodo(self) -> float:
        return 2.0 * np.pi / self.omega

    @property
    def amplitud_total(self) -> float:
        """R tal que a(t) = R sin(omega t + delta)."""
        return float(np.hypot(self.u0, self.v0 / self.omega))

    @property
    def fase(self) -> float:
        """delta tal que a(t) = R sin(omega t + delta)."""
        return float(np.arctan2(self.u0, self.v0 / self.omega))

    # ------------------------------------------------------------------ #
    # Solución analítica
    # ------------------------------------------------------------------ #
    def a_exacta(self, t):
        w = self.omega
        return self.u0 * np.cos(w * t) + (self.v0 / w) * np.sin(w * t)

    def da_exacta(self, t):
        w = self.omega
        return -self.u0 * w * np.sin(w * t) + self.v0 * np.cos(w * t)

    def d2a_exacta(self, t):
        return -self.kappa2 * self.a_exacta(t)

    @staticmethod
    def modo_espacial(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def campo(self, x, y, amplitud):
        """U(x, y, t) = phi(x, y) a(t) para un valor (o arreglo) de amplitud."""
        return self.modo_espacial(x, y) * amplitud

    def U_exacta(self, x, y, t):
        return self.campo(x, y, self.a_exacta(t))

    # ------------------------------------------------------------------ #
    # Energía mecánica
    # ------------------------------------------------------------------ #
    def energia_reducida(self, a, da):
        """E = (a'^2 + kappa^2 a^2) / 8, válida para U = phi(x, y) a(t)."""
        return 0.125 * (np.asarray(da) ** 2 + self.kappa2 * np.asarray(a) ** 2)

    @property
    def energia_inicial(self) -> float:
        return float(self.energia_reducida(self.u0, self.v0))

    def energia_cuadratura(self, a, da, n_malla: int = 101):
        """Energía integrando numéricamente (Simpson 2D) la densidad de energía.

        No usa la factorización de las integrales espaciales, por lo que
        sirve para validar la expresión reducida.
        """
        a = np.atleast_1d(a)
        da = np.atleast_1d(da)
        x = np.linspace(0.0, 1.0, n_malla)
        X, Y = np.meshgrid(x, x, indexing="ij")
        phi = self.modo_espacial(X, Y)
        phi_x = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
        phi_y = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)

        energias = np.empty(a.shape, dtype=float)
        for k, (ak, dak) in enumerate(zip(a, da)):
            densidad = 0.5 * ((phi * dak) ** 2 + (phi_x * ak) ** 2 + (phi_y * ak) ** 2)
            energias[k] = simpson(simpson(densidad, x=x, axis=1), x=x)
        return energias
