import numpy as np
import pytest

from pinn_membrana import ArquitecturaTemporal, MembranaCuadrada, ModeloA, ModeloB, PerdidaPINN

MEMBRANA = MembranaCuadrada()
T = np.linspace(0.0, 3.0, 50)


def test_solucion_exacta_cumple_edo_y_condiciones_iniciales():
    assert MEMBRANA.a_exacta(0.0) == pytest.approx(MEMBRANA.u0)
    assert MEMBRANA.da_exacta(0.0) == pytest.approx(MEMBRANA.v0)
    np.testing.assert_allclose(MEMBRANA.d2a_exacta(T) + MEMBRANA.kappa2 * MEMBRANA.a_exacta(T), 0, atol=1e-12)


def test_energia_reducida_igual_a_cuadratura():
    a, da = MEMBRANA.a_exacta(T[:5]), MEMBRANA.da_exacta(T[:5])
    np.testing.assert_allclose(MEMBRANA.energia_cuadratura(a, da), MEMBRANA.energia_reducida(a, da), rtol=1e-8)


@pytest.mark.parametrize("modelo", [ModeloA(), ModeloB()])
def test_derivadas_temporales_contra_diferencias_finitas(modelo):
    theta = modelo.muestrear_parametros(np.random.default_rng(3))
    a, da, d2a = modelo.derivadas(T, theta)
    h = 1e-5
    a_mas, da_mas, _ = modelo.derivadas(T + h, theta)
    a_men, da_men, _ = modelo.derivadas(T - h, theta)
    np.testing.assert_allclose(da, (a_mas - a_men) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(d2a, (da_mas - da_men) / (2 * h), atol=1e-5)


def test_jacobiano_analitico_modelo_A_igual_a_paso_complejo():
    modelo = ModeloA()
    theta = modelo.muestrear_parametros(np.random.default_rng(5))
    for J1, J2 in zip(modelo.jacobiano(T, theta), ArquitecturaTemporal.jacobiano(modelo, T, theta)):
        np.testing.assert_allclose(J1, J2, atol=1e-10)


@pytest.mark.parametrize("modelo", [ModeloA(), ModeloB()])
def test_gradiente_perdida_contra_diferencias_finitas(modelo):
    perdida = PerdidaPINN(modelo, MEMBRANA, T, 2.0, 3.0)
    theta = modelo.muestrear_parametros(np.random.default_rng(7))
    g = perdida.gradiente(theta)
    h = 1e-6
    g_fd = [(perdida(theta + h * e) - perdida(theta - h * e)) / (2 * h) for e in np.eye(theta.size)]
    np.testing.assert_allclose(g, g_fd, rtol=1e-6, atol=1e-6 * np.abs(g).max())


def test_solucion_exacta_anula_la_perdida_del_modelo_A():
    R, delta = MEMBRANA.amplitud_total, MEMBRANA.fase
    theta = np.array([R, MEMBRANA.omega, delta, 0.0, 1.0, 0.0])
    assert PerdidaPINN(ModeloA(), MEMBRANA, T)(theta) < 1e-24
