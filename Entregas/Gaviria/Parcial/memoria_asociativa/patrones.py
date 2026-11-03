"""Patrones binarios almacenados en la memoria asociativa."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# '#' = píxel activo (+1), '.' = píxel inactivo (-1)
LETRAS_8x8: dict[str, tuple[str, ...]] = {
    "J": (
        "########",
        "########",
        "....##..",
        "....##..",
        "....##..",
        "##..##..",
        "######..",
        ".####...",
    ),
    "H": (
        "##....##",
        "##....##",
        "##....##",
        "########",
        "########",
        "##....##",
        "##....##",
        "##....##",
    ),
}


@dataclass(frozen=True)
class Patron:
    nombre: str
    matriz: np.ndarray  # forma (alto, ancho), valores en {-1, +1}

    @classmethod
    def desde_texto(cls, nombre: str, filas: tuple[str, ...]) -> "Patron":
        matriz = np.array([[1 if c == "#" else -1 for c in fila] for fila in filas])
        return cls(nombre, matriz)

    @property
    def forma(self) -> tuple[int, int]:
        return self.matriz.shape

    @property
    def vector(self) -> np.ndarray:
        return self.matriz.ravel().astype(float)


class BancoPatrones:
    """Conjunto de P patrones organizados en la matriz X (P x N)."""

    def __init__(self, patrones: list[Patron]):
        formas = {p.forma for p in patrones}
        if len(formas) != 1:
            raise ValueError("Todos los patrones deben tener la misma forma.")
        self.patrones = list(patrones)
        self.forma = formas.pop()

    @classmethod
    def letras(cls, nombres: str = "JH") -> "BancoPatrones":
        return cls([Patron.desde_texto(n, LETRAS_8x8[n]) for n in nombres])

    # ------------------------------------------------------------------ #
    @property
    def X(self) -> np.ndarray:
        return np.vstack([p.vector for p in self.patrones])

    @property
    def P(self) -> int:
        return len(self.patrones)

    @property
    def N(self) -> int:
        return self.forma[0] * self.forma[1]

    @property
    def nombres(self) -> list[str]:
        return [p.nombre for p in self.patrones]

    def indice(self, nombre: str) -> int:
        return self.nombres.index(nombre)

    def __getitem__(self, clave) -> Patron:
        return self.patrones[self.indice(clave) if isinstance(clave, str) else clave]

    # ------------------------------------------------------------------ #
    def solapamientos(self) -> np.ndarray:
        """Matriz de Gram X X^T: (X X^T)_{mu nu} = x_mu . x_nu."""
        return self.X @ self.X.T

    def distancia_hamming(self, i, j) -> int:
        xi, xj = self[i].vector, self[j].vector
        return int(np.sum(xi != xj))

    def pixeles_distintos(self, i, j) -> np.ndarray:
        """Máscara booleana de los píxeles en que difieren los patrones i, j."""
        return self[i].vector != self[j].vector

    def a_imagen(self, q) -> np.ndarray:
        return np.asarray(q).reshape(self.forma)
