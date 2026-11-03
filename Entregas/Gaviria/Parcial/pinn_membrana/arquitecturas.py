"""Arquitecturas para la amplitud temporal a_theta(t) de la PINN.

Cada arquitectura entrega, de forma analítica, a(t), a'(t) y a''(t), que es
lo único que necesita el residuo físico. Las derivadas respecto a los
parámetros se obtienen por defecto con diferenciación de paso complejo
(exacta a precisión de máquina para funciones analíticas como sin y cos);
una arquitectura puede sobrescribir ``jacobiano`` con la expresión analítica.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

Derivadas = tuple[np.ndarray, np.ndarray, np.ndarray]


class ArquitecturaTemporal(ABC):
    """Interfaz común de las redes que representan a_theta(t)."""

    nombre: str = "arquitectura"

    @property
    @abstractmethod
    def nombres_parametros(self) -> tuple[str, ...]:
        ...

    @property
    def n_parametros(self) -> int:
        return len(self.nombres_parametros)

    @abstractmethod
    def derivadas(self, t, theta) -> Derivadas:
        """Devuelve (a, a', a'') evaluadas en t.

        Debe aceptar ``theta`` complejo para permitir el paso complejo.
        """

    @abstractmethod
    def muestrear_parametros(self, rng: np.random.Generator) -> np.ndarray:
        """Inicialización aleatoria de los parámetros."""

    # ------------------------------------------------------------------ #
    def __call__(self, t, theta):
        return self.derivadas(t, theta)[0]

    def jacobiano(self, t, theta, h: float = 1e-30) -> Derivadas:
        """Derivadas de (a, a', a'') respecto a theta por paso complejo.

        d f / d theta_k = Im f(theta + i h e_k) / h, sin error de cancelación.
        Cada arreglo devuelto tiene forma (len(t), n_parametros).
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        theta = np.asarray(theta, dtype=float)
        jac = [np.empty((t.size, self.n_parametros)) for _ in range(3)]
        for k in range(self.n_parametros):
            theta_c = theta.astype(complex)
            theta_c[k] += 1j * h
            for J, f in zip(jac, self.derivadas(t, theta_c)):
                J[:, k] = f.imag / h
        return tuple(jac)

    def describir(self, theta) -> dict[str, float]:
        return dict(zip(self.nombres_parametros, np.asarray(theta, dtype=float)))


@dataclass
class ModeloA(ArquitecturaTemporal):
    """Una capa de neuronas sinusoidales en paralelo.

    a(t) = sum_j A_j sin(omega_j t + b_j),
    theta = (A_1, omega_1, b_1, ..., A_n, omega_n, b_n).
    """

    n_neuronas: int = 2
    rango_amplitud: tuple[float, float] = (-2.0, 2.0)
    rango_omega: tuple[float, float] = (0.5, 10.0)
    nombre: str = "Modelo A"

    @property
    def nombres_parametros(self) -> tuple[str, ...]:
        return tuple(
            f"{p}_{j}" for j in range(1, self.n_neuronas + 1) for p in ("A", "omega", "b")
        )

    @staticmethod
    def _desempaquetar(theta):
        theta = np.asarray(theta)
        return theta[0::3], theta[1::3], theta[2::3]

    def derivadas(self, t, theta) -> Derivadas:
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        A, w, b = self._desempaquetar(theta)
        s, c = np.sin(w * t + b), np.cos(w * t + b)
        a = (A * s).sum(axis=1)
        da = (A * w * c).sum(axis=1)
        d2a = -(A * w**2 * s).sum(axis=1)
        return a, da, d2a

    def jacobiano(self, t, theta, h: float = 1e-30) -> Derivadas:
        """Jacobiano analítico de (a, a', a'') respecto a (A_j, omega_j, b_j)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        A, w, b = self._desempaquetar(np.asarray(theta, dtype=float))
        s, c = np.sin(w * t + b), np.cos(w * t + b)

        def intercalar(dA, dw, db):
            J = np.empty((t.shape[0], self.n_parametros))
            J[:, 0::3], J[:, 1::3], J[:, 2::3] = dA, dw, db
            return J

        Ja = intercalar(s, A * t * c, A * c)
        Jda = intercalar(w * c, A * c - A * w * t * s, -A * w * s)
        Jd2a = intercalar(-(w**2) * s, -2 * A * w * s - A * w**2 * t * c, -A * w**2 * c)
        return Ja, Jda, Jd2a

    def muestrear_parametros(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_neuronas
        theta = np.empty(3 * n)
        theta[0::3] = rng.uniform(*self.rango_amplitud, n)
        theta[1::3] = rng.uniform(*self.rango_omega, n)
        theta[2::3] = rng.uniform(-np.pi, np.pi, n)
        return theta


@dataclass
class ModeloB(ArquitecturaTemporal):
    """Dos capas consecutivas con una neurona sinusoidal cada una.

    h_1(t) = sin(omega_1 t + b_1),   a(t) = A sin(omega_2 h_1(t) + b_2),
    theta = (omega_1, b_1, omega_2, b_2, A).
    """

    rango_omega1: tuple[float, float] = (0.5, 10.0)
    rango_omega2: tuple[float, float] = (-3.0, 3.0)
    rango_amplitud: tuple[float, float] = (-2.0, 2.0)
    nombre: str = "Modelo B"

    @property
    def nombres_parametros(self) -> tuple[str, ...]:
        return ("omega_1", "b_1", "omega_2", "b_2", "A")

    def capa_oculta(self, t, theta):
        """(h_1, h_1', h_1'') con h_1'' = -omega_1^2 h_1."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w1, b1 = theta[0], theta[1]
        h = np.sin(w1 * t + b1)
        dh = w1 * np.cos(w1 * t + b1)
        return h, dh, -(w1**2) * h

    def derivadas(self, t, theta) -> Derivadas:
        theta = np.asarray(theta)
        _, _, w2, b2, A = theta
        h, dh, d2h = self.capa_oculta(t, theta)
        z = w2 * h + b2
        sz, cz = np.sin(z), np.cos(z)
        a = A * sz
        da = A * w2 * dh * cz
        d2a = A * (w2 * d2h * cz - (w2 * dh) ** 2 * sz)
        return a, da, d2a

    def muestrear_parametros(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [
                rng.uniform(*self.rango_omega1),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(*self.rango_omega2),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(*self.rango_amplitud),
            ]
        )
