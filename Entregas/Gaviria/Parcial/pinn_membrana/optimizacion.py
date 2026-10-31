"""Optimización de la pérdida de la PINN.

Se usa la estrategia habitual en PINNs: una fase de Adam (descenso de
gradiente estocástico-adaptativo, robusto lejos del mínimo) seguida de un
refinamiento cuasi-Newton con L-BFGS. Como la pérdida no es convexa en las
frecuencias, el entrenamiento se repite desde varias inicializaciones
aleatorias (multi-inicio) y se conserva la de menor pérdida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .perdida import PerdidaPINN


@dataclass
class Adam:
    tasa: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def minimizar(self, valor_y_gradiente, theta0, n_iter: int):
        theta = np.array(theta0, dtype=float)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        historia = []
        for k in range(1, n_iter + 1):
            L, g = valor_y_gradiente(theta)
            historia.append(L)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**k)
            v_hat = v / (1 - self.beta2**k)
            theta -= self.tasa * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta, np.array(historia)


@dataclass
class ResultadoEntrenamiento:
    theta_inicial: np.ndarray
    theta: np.ndarray
    perdida: float
    historia: np.ndarray
    iter_adam: int
    semilla: int | None = None

    @property
    def iter_totales(self) -> int:
        return self.historia.size


@dataclass
class EntrenadorPINN:
    perdida: PerdidaPINN
    adam: Adam = field(default_factory=Adam)
    iter_adam: int = 2000
    iter_lbfgs: int = 5000
    tol_lbfgs: float = 1e-15

    def entrenar(self, theta0, semilla: int | None = None) -> ResultadoEntrenamiento:
        theta0 = np.asarray(theta0, dtype=float)
        theta, hist_adam = self.adam.minimizar(
            self.perdida.valor_y_gradiente, theta0, self.iter_adam
        )

        hist_lbfgs: list[float] = []
        if self.iter_lbfgs > 0:
            res = minimize(
                self.perdida.valor_y_gradiente,
                theta,
                jac=True,
                method="L-BFGS-B",
                callback=lambda xk: hist_lbfgs.append(self.perdida(xk)),
                options={"maxiter": self.iter_lbfgs, "ftol": self.tol_lbfgs, "gtol": 1e-12},
            )
            theta = res.x

        historia = np.concatenate([hist_adam, hist_lbfgs])
        return ResultadoEntrenamiento(
            theta_inicial=theta0,
            theta=theta,
            perdida=self.perdida(theta),
            historia=historia,
            iter_adam=hist_adam.size,
            semilla=semilla,
        )

    def multiinicio(self, n_inicios: int, semilla: int = 0) -> list[ResultadoEntrenamiento]:
        """Entrena desde ``n_inicios`` puntos aleatorios; ordena por pérdida."""
        semillas = np.random.SeedSequence(semilla).generate_state(n_inicios)
        resultados = []
        for s in semillas:
            rng = np.random.default_rng(int(s))
            theta0 = self.perdida.modelo.muestrear_parametros(rng)
            resultados.append(self.entrenar(theta0, semilla=int(s)))
        return sorted(resultados, key=lambda r: r.perdida)
