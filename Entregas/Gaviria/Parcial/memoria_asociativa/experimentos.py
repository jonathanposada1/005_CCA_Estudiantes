"""Experimentos de recuperación y reducción exacta de la dinámica para dos patrones."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import hypergeom

from .patrones import BancoPatrones
from .red import RedMemoriaAsociativa, Trayectoria, precision


def perturbar(x, fraccion: float, rng: np.random.Generator) -> np.ndarray:
    """Invierte exactamente round(fraccion * N) píxeles escogidos al azar."""
    x = np.asarray(x, dtype=float).copy()
    n = int(round(fraccion * x.size))
    idx = rng.choice(x.size, size=n, replace=False)
    x[idx] *= -1
    return x


# ---------------------------------------------------------------------- #
# Teoría para P = 2
# ---------------------------------------------------------------------- #
@dataclass
class TeoriaDosPatrones:
    """Reducción exacta de la dinámica (alpha = 1) para dos patrones.

    Tras la primera iteración, q^{(t)} = (1+u_t)/2 x_1 + (1-u_t)/2 x_2 y

        u_{t+1} = tanh(beta d u_t),   u_1 = tanh(beta (d - 2k)),

    con d la distancia de Hamming entre los patrones y k el número de
    píxeles invertidos dentro de los d píxeles en que difieren.
    """

    banco: BancoPatrones
    objetivo: str
    otro: str

    def __post_init__(self):
        self.x1 = self.banco[self.objetivo].vector
        self.x2 = self.banco[self.otro].vector
        self.d = self.banco.distancia_hamming(self.objetivo, self.otro)
        self.N = self.banco.N

    @property
    def beta_critico(self) -> float:
        return 1.0 / self.d

    # Dinámica reducida ------------------------------------------------- #
    def u_de_estado(self, q) -> float:
        """Coordenada u tal que (x1 - x2).q = 2 d u."""
        return float((self.x1 - self.x2) @ q / (2 * self.d))

    def estado(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)[..., None]
        return 0.5 * (1 + u) * self.x1 + 0.5 * (1 - u) * self.x2

    def mapa(self, u, beta):
        return np.tanh(beta * self.d * np.asarray(u))

    def punto_fijo(self, beta: float) -> float:
        """Raíz positiva de u = tanh(beta d u); vale 0 si beta <= beta_c."""
        if beta * self.d <= 1:
            return 0.0
        return brentq(lambda u: u - np.tanh(beta * self.d * u), 1e-12, 1.0)

    def tasa_convergencia(self, beta: float) -> float:
        """lambda = |f'(u*)| = beta d (1 - u*^2): |u_{t+1} - u*| ~ lambda |u_t - u*|."""
        u = self.punto_fijo(beta)
        return beta * self.d * (1 - u**2)

    def iteraciones_teoricas(self, beta: float, u0: float, tol: float = 1e-10,
                             max_iter: int = 100_000) -> int:
        """Iteraciones del mapa reducido hasta |u_{t+1} - u_t| < tol."""
        u, n = u0, 0
        while n < max_iter:
            u_nuevo = self.mapa(u, beta)
            n += 1
            if abs(u_nuevo - u) < tol:
                break
            u = u_nuevo
        return n

    # Estadística de la perturbación ---------------------------------- #
    def distribucion_k(self, fraccion: float):
        """k ~ Hipergeométrica(N, d, n): inversiones que caen donde difieren."""
        n = int(round(fraccion * self.N))
        return hypergeom(self.N, self.d, n)

    def probabilidades(self, fraccion: float) -> dict[str, float]:
        dist = self.distribucion_k(fraccion)
        mitad = self.d / 2
        p_obj = dist.cdf(np.ceil(mitad) - 1)       # k < d/2
        p_otro = dist.sf(np.floor(mitad))          # k > d/2
        return {"objetivo": float(p_obj), "otro": float(p_otro),
                "empate": float(1 - p_obj - p_otro)}

    def precision_esperada(self, fraccion: float) -> float:
        """Precisión binaria media predicha (independiente de beta)."""
        pr = self.probabilidades(fraccion)
        prec_otro = np.mean(self.x2 == self.x1)
        prec_empate = precision(self.estado(0.0), self.x1)
        return pr["objetivo"] + pr["otro"] * prec_otro + pr["empate"] * prec_empate


# ---------------------------------------------------------------------- #
# Barridos
# ---------------------------------------------------------------------- #
@dataclass
class ResultadoBarridoRuido:
    fracciones: np.ndarray
    beta: float
    precision: np.ndarray     # (n_fracciones, n_repeticiones)
    recuperado: np.ndarray    # índice del patrón recuperado (-1: ninguno)
    iteraciones: np.ndarray

    def tasa(self, indice: int) -> np.ndarray:
        """Fracción de repeticiones que convergen al patrón ``indice`` (-1: ninguno)."""
        return np.mean(self.recuperado == indice, axis=1)

    @property
    def n_repeticiones(self) -> int:
        return self.precision.shape[1]

    def intervalo_media(self, valores: np.ndarray, z: float = 1.96) -> np.ndarray:
        """Semiancho del intervalo de confianza (95 %) de la media por fila."""
        return z * valores.std(axis=1, ddof=1) / np.sqrt(self.n_repeticiones)

    @property
    def precision_media(self) -> np.ndarray:
        return self.precision.mean(axis=1)


def barrido_ruido(red: RedMemoriaAsociativa, banco: BancoPatrones, objetivo: str,
                  fracciones: Sequence[float], n_repeticiones: int,
                  semilla: int = 0, **kw_recuperar) -> ResultadoBarridoRuido:
    rng = np.random.default_rng(semilla)
    x = banco[objetivo].vector
    forma = (len(fracciones), n_repeticiones)
    prec = np.empty(forma)
    recup, iters = np.empty(forma, dtype=int), np.empty(forma, dtype=int)
    for i, f in enumerate(fracciones):
        for j in range(n_repeticiones):
            tray = red.recuperar(perturbar(x, f, rng), **kw_recuperar)
            prec[i, j] = precision(tray.final, x)
            recup[i, j] = red.patron_recuperado(tray.final)
            iters[i, j] = tray.n_iter
    return ResultadoBarridoRuido(np.asarray(fracciones), red.beta, prec, recup, iters)


def barrido_beta(X, q0, betas: Sequence[float], **kw_recuperar) -> dict[float, Trayectoria]:
    """Recupera desde el mismo estado inicial para distintos valores de beta."""
    return {b: RedMemoriaAsociativa(X, b).recuperar(q0, **kw_recuperar) for b in betas}
