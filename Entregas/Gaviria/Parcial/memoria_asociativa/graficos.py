"""Gráficas de la red de memoria asociativa."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .experimentos import ResultadoBarridoRuido, TeoriaDosPatrones
from .patrones import BancoPatrones
from .red import Trayectoria, binarizar


def _imagen(ax, banco: BancoPatrones, q, titulo, cmap="gray_r", vmin=-1, vmax=1):
    im = ax.imshow(banco.a_imagen(q), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def mostrar_patrones(banco: BancoPatrones):
    fig, axes = plt.subplots(1, banco.P, figsize=(2.6 * banco.P, 2.8))
    for ax, p in zip(np.atleast_1d(axes), banco.patrones):
        _imagen(ax, banco, p.vector, f"Patrón {p.nombre}")
    fig.tight_layout()
    return fig


def comparar_recuperacion(banco: BancoPatrones, objetivo: str, tray: Trayectoria,
                          titulo: str = ""):
    x = banco[objetivo].vector
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout="constrained")
    _imagen(axes[0], banco, x, f"Original ({objetivo})")
    _imagen(axes[1], banco, tray.inicial,
            f"Perturbado ({np.mean(tray.inicial != x):.0%} invertidos)")
    im = _imagen(axes[2], banco, tray.final, "Estado final continuo", cmap="RdBu_r")
    acierto = np.mean(tray.final_binario == x)
    _imagen(axes[3], banco, tray.final_binario, f"Final binario ({acierto:.1%})")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label=r"$q_i$")
    if titulo:
        fig.suptitle(titulo)
    return fig


def curva_recuperacion(res: ResultadoBarridoRuido, teoria: TeoriaDosPatrones):
    """Calidad de la recuperación frente al porcentaje de píxeles invertidos.

    Puntos: media de las repeticiones; banda: intervalo de confianza del 95 % de
    la media; líneas: teoría, evaluada en los mismos porcentajes (el número de
    píxeles invertidos es entero, n = f N).
    """
    pct = 100 * res.fracciones
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    media, ic = 100 * res.precision_media, 100 * res.intervalo_media(res.precision)
    axes[0].fill_between(pct, media - ic, media + ic, color="tab:blue", alpha=0.25,
                         label="IC 95 % de la media")
    axes[0].plot(pct, media, "o", ms=3.5, color="tab:blue", label="Simulación")
    axes[0].plot(pct, [100 * teoria.precision_esperada(f) for f in res.fracciones], "k-",
                 lw=1.2, label="Teoría")
    axes[0].set_ylabel("Píxeles que coinciden con el original (%)")
    axes[0].set_title(rf"Precisión binaria ($\beta={res.beta:g}$)")

    pr = [teoria.probabilidades(f) for f in res.fracciones]
    casos = [(teoria.banco.indice(teoria.objetivo), "objetivo", teoria.objetivo, "tab:blue"),
             (teoria.banco.indice(teoria.otro), "otro", teoria.otro, "tab:red"),
             (-1, "empate", "mezcla (empate)", "tab:gray")]
    for indice, clave, nombre, color in casos:
        axes[1].plot(pct, res.tasa(indice), "o", ms=3.5, color=color, label=f"Simulación: {nombre}")
        axes[1].plot(pct, [p[clave] for p in pr], "-", color=color, lw=1.2)
    axes[1].plot([], [], "k-", lw=1.2, label="Teoría")
    axes[1].set_ylabel("Probabilidad")
    axes[1].set_title(rf"Estado final al que converge la red ($\beta={res.beta:g}$)")

    for ax, lugar in zip(axes, ["lower left", "center left"]):
        ax.set_xlabel("Píxeles invertidos (%)")
        ax.legend(fontsize=8, loc=lugar)
    fig.tight_layout()
    return fig


def evolucion_energia(trayectorias: Mapping[float, Trayectoria], objetivo_idx: int = 0):
    """Izquierda: E(q^(t)) - E(q^*) en escala logarítmica.

    Se omiten los puntos por debajo del piso de redondeo de E (~50 eps |E|):
    ahí la diferencia ya no mide la convergencia sino el error de punto
    flotante. Con eje t lineal, la convergencia geométrica es una recta.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for beta, tr in trayectorias.items():
        dE, visibles = tr.brecha_energia()
        it = np.arange(dE.size)
        etiqueta = rf"$\beta={beta:.3g}$"
        linea, = axes[0].semilogy(it[visibles], dE[visibles], "o-", ms=3, label=etiqueta)
        axes[1].plot(it, tr.pesos[:, objetivo_idx], "o-", ms=3, color=linea.get_color(),
                     label=etiqueta)
    axes[0].set_xlabel("Iteración $t$")
    axes[0].set_ylabel(r"$E(\mathbf{q}^{(t)})-E(\mathbf{q}^{*})$")
    axes[0].set_title("Evolución de la energía (hasta el piso de redondeo)")
    axes[1].set_xlabel("Iteración $t$")
    axes[1].set_ylabel(r"$p_{\rm obj}^{(t)}=\mathrm{softmax}(\beta X\mathbf{q}^{(t)})_{\rm obj}$")
    axes[1].set_title("Peso del patrón objetivo")
    axes[1].axhline(0.5, color="0.5", ls=":")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def estados_finales(banco: BancoPatrones, trayectorias: Mapping[float, Trayectoria]):
    n = len(trayectorias)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), layout="constrained")
    for j, (beta, tr) in enumerate(trayectorias.items()):
        im = _imagen(axes[0, j], banco, tr.final, rf"$\beta={beta:.3g}$", cmap="RdBu_r")
        _imagen(axes[1, j], banco, binarizar(tr.final), "binario")
    fig.colorbar(im, ax=axes[0, :], fraction=0.02, label=r"$q_i^{*}$")
    return fig


def bifurcacion(teoria: TeoriaDosPatrones, betas, u_sim, iter_sim, u0: float,
                tol: float = 1e-10, max_iter: int = 100_000):
    b_fino = np.geomspace(min(betas), max(betas), 400)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    u_teo = np.array([teoria.punto_fijo(b) for b in b_fino])
    axes[0].plot(b_fino, u_teo, "k-", label=r"Teoría: $u^*=\tanh(\beta d\,u^*)$")
    axes[0].plot(b_fino, -u_teo, "k--", lw=0.8)
    axes[0].plot(betas, u_sim, "o", color="tab:blue", label="Simulación")
    axes[1].plot(b_fino, [teoria.iteraciones_teoricas(b, u0, tol, max_iter) for b in b_fino], "k-",
                 label="Mapa reducido")
    axes[1].plot(betas, iter_sim, "o", color="tab:blue", label="Simulación")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.axvline(teoria.beta_critico, color="tab:red", ls=":",
                   label=rf"$\beta_c=1/d={teoria.beta_critico:.4f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$u^*$")
    axes[0].set_title("Diagrama de bifurcación del estado final")
    axes[1].set_ylabel("Iteraciones hasta convergencia")
    axes[1].set_title("Ralentización crítica")
    fig.tight_layout()
    return fig

