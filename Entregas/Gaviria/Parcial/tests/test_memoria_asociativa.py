import numpy as np
import pytest

from memoria_asociativa import BancoPatrones, RedMemoriaAsociativa, TeoriaDosPatrones, perturbar

BANCO = BancoPatrones.letras("JH")


def test_matriz_de_patrones():
    assert BANCO.X.shape == (2, 64)
    assert set(np.unique(BANCO.X)) == {-1.0, 1.0}


@pytest.mark.parametrize("beta", [0.01, 0.1, 5.0])
def test_gradiente_contra_diferencias_finitas(beta):
    red = RedMemoriaAsociativa(BANCO.X, beta)
    q = np.random.default_rng(0).normal(size=64)
    h = 1e-6
    g_fd = [(red.energia(q + h * e) - red.energia(q - h * e)) / (2 * h) for e in np.eye(64)]
    np.testing.assert_allclose(red.gradiente(q), g_fd, atol=1e-6)


@pytest.mark.parametrize("beta", [0.005, 0.05, 1.0])
def test_energia_no_aumenta_con_alpha_1(beta):
    red = RedMemoriaAsociativa(BANCO.X, beta)
    q0 = perturbar(BANCO["J"].vector, 0.3, np.random.default_rng(1))
    assert np.all(np.diff(red.recuperar(q0, max_iter=500).energias) <= 1e-12)


@pytest.mark.parametrize("beta", [0.01, 0.05, 0.2])
def test_dinamica_reducida_coincide_con_la_red(beta):
    teoria = TeoriaDosPatrones(BANCO, "J", "H")
    red = RedMemoriaAsociativa(BANCO.X, beta)
    q0 = perturbar(BANCO["J"].vector, 0.25, np.random.default_rng(2))
    tray = red.recuperar(q0, max_iter=50)
    u = teoria.u_de_estado(q0)
    for q in tray.estados[1:]:
        u = teoria.mapa(u, beta)
        np.testing.assert_allclose(q, teoria.estado(u), atol=1e-12)
