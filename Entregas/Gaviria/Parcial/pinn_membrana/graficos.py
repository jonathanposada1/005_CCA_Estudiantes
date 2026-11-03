"""Gráficas del análisis de la PINN de la membrana."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .analisis import EvaluadorSolucion
from .fisica import MembranaCuadrada
from .optimizacion import ResultadoEntrenamiento

COLORES = {"Exacta": "k", "Modelo A": "tab:blue", "Modelo B": "tab:red"}


def _color(nombre: str) -> str:
    return COLORES.get(nombre, None)


def _sombrear_entrenamiento(ax, T, etiqueta: bool = True):
    ax.axvspan(0, T, color="0.92", zorder=0,
               label="Ventana de colocación" if etiqueta else None)


def _espacio_leyenda(ax, fraccion: float = 0.45):
    """Amplía el límite superior del eje para que la leyenda no tape las curvas."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + fraccion * (y1 - y0))


def _media_movil_log(historia, n_adam: int, ventana: int = 31):
    """Media móvil centrada de log10 L, calculada por separado en la fase de Adam
    y en la de L-BFGS para no mezclar dos dinámicas distintas."""
    tramos = []
    for tramo in (historia[:n_adam], historia[n_adam:]):
        if tramo.size == 0:
            continue
        w = min(ventana, max(1, (tramo.size // 2) * 2 - 1))
        relleno = np.pad(np.log10(tramo), w // 2, mode="edge")
        tramos.append(np.convolve(relleno, np.ones(w) / w, mode="valid"))
    return 10 ** np.concatenate(tramos)


def historias_perdida(resultados: Mapping[str, ResultadoEntrenamiento],
                      perdida_trivial: float | None = None, ventana: int = 31):
    """Curva tenue: pérdida en cada iteración. Curva gruesa: media móvil de
    log10 L en ``ventana`` iteraciones, que filtra las caídas puntuales de Adam
    al cruzar el mínimo sin ocultar la tendencia. La línea punteada vertical
    marca el paso de Adam a L-BFGS; la discontinua, la pérdida de a_theta = 0."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for nombre, res in resultados.items():
        ax.semilogy(res.historia, color=_color(nombre), lw=0.7, alpha=0.3)
        ax.semilogy(_media_movil_log(res.historia, res.iter_adam, ventana), color=_color(nombre),
                    lw=2, label=f"{nombre} (media móvil de $\\log_{{10}}\\mathcal{{L}}$, {ventana} it.)")
        ax.axvline(res.iter_adam, color=_color(nombre), ls=":", lw=1)
    if perdida_trivial is not None:
        ax.axhline(perdida_trivial, color="0.4", ls="--", lw=1,
                   label=r"Solución trivial $a_\theta\equiv0$: $\lambda_0u_0^2+\lambda_1v_0^2$")
    ax.set_xlabel("Iteración (Adam  |  L-BFGS)")
    ax.set_ylabel(r"$\mathcal{L}(\theta)$")
    ax.set_title("Historia de la pérdida del mejor inicio")
    ax.legend()
    fig.tight_layout()
    return fig


def comparar_amplitudes(t, membrana: MembranaCuadrada,
                        evaluadores: Mapping[str, EvaluadorSolucion], T_entrenamiento):
    """Arriba: a_theta(t) frente a a(t). Abajo: error con signo e(t)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    _sombrear_entrenamiento(ax1, T_entrenamiento)
    _sombrear_entrenamiento(ax2, T_entrenamiento, etiqueta=False)
    ax1.plot(t, membrana.a_exacta(t), "k-", lw=3, alpha=0.35, label="Exacta")
    for nombre, ev in evaluadores.items():
        e = ev.error_amplitud(t)
        ax1.plot(t, ev.modelo(t, ev.theta), "--", color=_color(nombre), lw=1.6, label=nombre)
        ax2.plot(t, e, color=_color(nombre), lw=1.2,
                 label=rf"{nombre}  ($\max|e|={np.max(np.abs(e)):.1e}$)")
    ax1.set_ylabel(r"$a(t)$")
    ax1.set_title(r"Amplitud modal: PINN vs solución analítica")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel(r"$e(t)=a_\theta(t)-a(t)$")
    ax2.set_xlabel(r"$t$")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    _espacio_leyenda(ax1, 0.2)
    _espacio_leyenda(ax2)
    ax1.legend(loc="upper right", ncol=4, fontsize=9)
    ax2.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def mapas_campo(evaluadores: Mapping[str, EvaluadorSolucion], tiempos, n: int = 81):
    """U(x, y, t) exacto (primera fila) y de cada modelo, con escala común.

    El título de cada panel de un modelo da max |U_theta - U| en ese instante,
    evaluado en forma factorizada phi(x, y) e(t)."""
    m = next(iter(evaluadores.values())).membrana
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    filas = len(evaluadores) + 1
    fig, axes = plt.subplots(filas, len(tiempos), figsize=(2.9 * len(tiempos), 2.9 * filas),
                             layout="constrained")
    estilo = dict(cmap="RdBu_r", vmin=-m.amplitud_total, vmax=m.amplitud_total, shading="auto")
    for j, t in enumerate(tiempos):
        axes[0, j].pcolormesh(X, Y, m.U_exacta(X, Y, t), **estilo)
        axes[0, j].set_title(f"Exacta, $t={t:.2f}$", fontsize=10)
        for i, (nombre, ev) in enumerate(evaluadores.items(), start=1):
            axes[i, j].pcolormesh(X, Y, ev.campo(X, Y, t), **estilo)
            err = np.abs(ev.error_campo(X, Y, t)).max()
            axes[i, j].set_title(rf"$\max|U_\theta-U|={err:.1e}$", fontsize=9)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for i, nombre in enumerate(["Exacta", *evaluadores], start=0):
        axes[i, 0].set_ylabel(nombre)
    fig.colorbar(axes[0, 0].collections[0], ax=axes, fraction=0.02, label=r"$U(x,y,t)$")
    return fig


def energia(t, evaluadores: Mapping[str, EvaluadorSolucion], T_entrenamiento,
            modelo_potencia: str = "Modelo B"):
    """Izquierda: E_theta/E_0 - 1 de todos los modelos (misma escala lineal).

    Derecha: verificación de dE/dt = a' r / 4 para ``modelo_potencia``. Sólo se
    muestra un modelo con residuo apreciable: si r ~ 0, la derivada numérica de
    E es puro error de redondeo y la comparación no tiene contenido.
    """
    m = next(iter(evaluadores.values())).membrana
    E0 = m.energia_inicial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    _sombrear_entrenamiento(ax1, T_entrenamiento)
    _sombrear_entrenamiento(ax2, T_entrenamiento, etiqueta=False)
    for nombre, ev in evaluadores.items():
        dE = ev.energia(t) / E0 - 1
        ax1.plot(t, dE, color=_color(nombre), lw=1.4,
                 label=rf"{nombre}  ($\max|\Delta E|/E_0={np.max(np.abs(dE)):.1e}$)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel(r"$E_\theta(t)/E_0-1$")
    ax1.set_title("Desviación relativa de la energía mecánica")
    _espacio_leyenda(ax1)
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    ev = evaluadores[modelo_potencia]
    ax2.plot(t, ev.potencia(t), color=_color(modelo_potencia), lw=2, alpha=0.6,
             label=r"$\frac{1}{4}a_\theta'\,r_\theta$ (identidad analítica)")
    ax2.plot(t, np.gradient(ev.energia(t), t), "k:", lw=1.2,
             label=r"$dE_\theta/dt$ (derivada numérica)")
    ax2.set_title(f"{modelo_potencia}: potencia $dE_\\theta/dt$")
    _espacio_leyenda(ax2, 0.3)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$t$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.tight_layout()
    return fig
