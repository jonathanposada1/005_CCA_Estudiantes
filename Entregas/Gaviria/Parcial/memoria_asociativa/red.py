"""Red de memoria asociativa con energía log-sum-exp (Hopfield moderna).

    E(q) = 1/2 ||q||^2 - (1/beta) ln sum_mu exp(beta x_mu . q)

    grad E(q) = q - X^T softmax(beta X q)

    q^{(t+1)} = q^{(t)} - alpha grad E(q^{(t)})
              = (1 - alpha) q^{(t)} + alpha X^T softmax(beta X q^{(t)})
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def binarizar(q) -> np.ndarray:
    """Patrón binario según el signo; por convención sign(0) = +1."""
    return np.where(np.asarray(q) >= 0, 1.0, -1.0)


def precision(q, patron) -> float:
    """Fracción de píxeles de sign(q) que coinciden con el patrón."""
    return float(np.mean(binarizar(q) == np.asarray(patron)))


@dataclass
class Trayectoria:
    estados: np.ndarray   # (T + 1, N)
    energias: np.ndarray  # (T + 1,)
    pesos: np.ndarray     # (T + 1, P), softmax(beta X q^{(t)})
    convergio: bool

    @property
    def n_iter(self) -> int:
        return self.estados.shape[0] - 1

    @property
    def inicial(self) -> np.ndarray:
        return self.estados[0]

    @property
    def final(self) -> np.ndarray:
        return self.estados[-1]

    @property
    def final_binario(self) -> np.ndarray:
        return binarizar(self.final)

    def brecha_energia(self) -> tuple[np.ndarray, np.ndarray]:
        """E(q^(t)) - E(q^*) y la máscara de los valores por encima del piso de
        redondeo de E (~50 eps |E|), donde la brecha todavía mide convergencia."""
        dE = self.energias - self.energias[-1]
        piso = 50 * np.finfo(float).eps * np.abs(self.energias).max()
        return dE, dE > piso

    def cociente_energia(self) -> float:
        """Cociente asintótico [E(q^(t+1)) - E*] / [E(q^(t)) - E*] (mediana de la
        segunda mitad de los pasos resolubles); nan si hay muy pocos."""
        dE, visibles = self.brecha_energia()
        v = dE[1:][visibles[1:]]  # t = 0 no está sobre la recta J-H
        if v.size < 4:
            return float("nan")
        cocientes = v[1:] / v[:-1]
        return float(np.median(cocientes[cocientes.size // 2:]))


@dataclass
class RedMemoriaAsociativa:
    X: np.ndarray
    beta: float = 1.0

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.beta <= 0:
            raise ValueError("beta debe ser positivo.")

    @property
    def P(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    # ------------------------------------------------------------------ #
    def similitudes(self, q) -> np.ndarray:
        """m_mu = sum_i x_{mu i} q_i."""
        return self.X @ q

    def _pesos_y_energia(self, q) -> tuple[np.ndarray, float]:
        """softmax(beta X q) y E(q) a partir de un único producto X q.

        Se resta el máximo de beta m antes de exponenciar (estabilidad numérica):
        ln sum exp(z) = z_max + ln sum exp(z - z_max).
        """
        z = self.beta * self.similitudes(q)
        z_max = z.max()
        e = np.exp(z - z_max)
        suma = e.sum()
        energia = 0.5 * q @ q - (z_max + np.log(suma)) / self.beta
        return e / suma, float(energia)

    def pesos(self, q) -> np.ndarray:
        """p_mu = softmax(beta m)_mu (implementación numéricamente estable)."""
        return self._pesos_y_energia(q)[0]

    def energia(self, q) -> float:
        return self._pesos_y_energia(q)[1]

    def gradiente(self, q) -> np.ndarray:
        """grad E(q) = q - X^T softmax(beta X q)."""
        return q - self.X.T @ self.pesos(q)

    # ------------------------------------------------------------------ #
    def recuperar(self, q0, alpha: float = 1.0, tol: float = 1e-10,
                  max_iter: int = 200) -> Trayectoria:
        """Itera la regla de actualización hasta ||q^{(t+1)} - q^{(t)}||_inf < tol."""
        q = np.asarray(q0, dtype=float)
        p, E = self._pesos_y_energia(q)
        estados, energias, pesos = [q], [E], [p]
        convergio = False
        for _ in range(max_iter):
            q_nuevo = (1 - alpha) * q + alpha * (self.X.T @ p)  # = q - alpha grad E(q)
            p, E = self._pesos_y_energia(q_nuevo)
            estados.append(q_nuevo)
            energias.append(E)
            pesos.append(p)
            if np.max(np.abs(q_nuevo - q)) < tol:
                convergio = True
                break
            q = q_nuevo
        return Trayectoria(np.array(estados), np.array(energias), np.array(pesos), convergio)

    def patron_recuperado(self, q) -> int:
        """Índice del patrón que coincide exactamente con sign(q); -1 si ninguno."""
        coincide = np.all(self.X == binarizar(q), axis=1)
        return int(np.argmax(coincide)) if coincide.any() else -1
