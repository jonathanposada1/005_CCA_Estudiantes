"""Residuo físico y función de pérdida de la PINN.

La pérdida se escribe como un problema de mínimos cuadrados no lineales,

    L(theta) = || R(theta) ||^2,

    R(theta) = [ r(t_1)/sqrt(N), ..., r(t_N)/sqrt(N),
                 sqrt(lambda_0) (a(0) - u0),
                 sqrt(lambda_1) (a'(0) - v0) ],

con r(t) = a''(t) + kappa^2 a(t). Así, el gradiente es 2 J^T R, con J el
jacobiano de R, y la misma estructura sirve para métodos tipo Gauss-Newton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arquitecturas import ArquitecturaTemporal
from .fisica import MembranaCuadrada


@dataclass
class PerdidaPINN:
    modelo: ArquitecturaTemporal
    membrana: MembranaCuadrada
    t_colocacion: np.ndarray
    lambda_0: float = 1.0
    lambda_1: float = 1.0

    def __post_init__(self):
        self.t_colocacion = np.asarray(self.t_colocacion, dtype=float)
        self._t0 = np.zeros(1)

    @property
    def n_colocacion(self) -> int:
        return self.t_colocacion.size

    @property
    def perdida_trivial(self) -> float:
        """Pérdida de a_theta = 0: residuo nulo, sólo pagan las condiciones iniciales."""
        return self.lambda_0 * self.membrana.u0**2 + self.lambda_1 * self.membrana.v0**2

    # ------------------------------------------------------------------ #
    # Residuos
    # ------------------------------------------------------------------ #
    def residuo_fisico(self, t, theta):
        """r(t) = a''(t) + kappa^2 a(t)."""
        a, _, d2a = self.modelo.derivadas(t, theta)
        return d2a + self.membrana.kappa2 * a

    def residuos_iniciales(self, theta):
        a0, da0, _ = self.modelo.derivadas(self._t0, theta)
        return a0[0] - self.membrana.u0, da0[0] - self.membrana.v0

    def vector_residuos(self, theta) -> np.ndarray:
        r = self.residuo_fisico(self.t_colocacion, theta) / np.sqrt(self.n_colocacion)
        e0, e1 = self.residuos_iniciales(theta)
        return np.concatenate([r, [np.sqrt(self.lambda_0) * e0, np.sqrt(self.lambda_1) * e1]])

    def jacobiano(self, theta) -> np.ndarray:
        """Jacobiano de R(theta), de forma (N + 2, n_parametros)."""
        Ja, _, Jd2a = self.modelo.jacobiano(self.t_colocacion, theta)
        Jr = (Jd2a + self.membrana.kappa2 * Ja) / np.sqrt(self.n_colocacion)
        Ja0, Jda0, _ = self.modelo.jacobiano(self._t0, theta)
        return np.vstack([Jr, np.sqrt(self.lambda_0) * Ja0, np.sqrt(self.lambda_1) * Jda0])

    # ------------------------------------------------------------------ #
    # Pérdida y gradiente
    # ------------------------------------------------------------------ #
    def __call__(self, theta) -> float:
        R = self.vector_residuos(theta)
        return float(R @ R)

    def gradiente(self, theta) -> np.ndarray:
        return 2.0 * self.jacobiano(theta).T @ self.vector_residuos(theta)

    def valor_y_gradiente(self, theta):
        R = self.vector_residuos(theta)
        return float(R @ R), 2.0 * self.jacobiano(theta).T @ R

    def componentes(self, theta) -> dict[str, float]:
        """Contribución de cada término a la pérdida total."""
        r = self.residuo_fisico(self.t_colocacion, theta)
        e0, e1 = self.residuos_iniciales(theta)
        fisica = float(np.mean(r**2))
        ci_pos = float(self.lambda_0 * e0**2)
        ci_vel = float(self.lambda_1 * e1**2)
        return {
            "fisica": fisica,
            "ci_posicion": ci_pos,
            "ci_velocidad": ci_vel,
            "total": fisica + ci_pos + ci_vel,
        }
