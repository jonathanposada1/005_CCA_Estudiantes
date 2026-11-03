"""
Genera 3 datasets sinteticos de experimentos clasicos de fisica, con ruido,
para practicar ajuste de curvas (curve fitting) con el agente.

Parametros "verdaderos" usados para generar cada dataset (respuesta de
referencia para el profesor, el agente NO los conoce):

- resorte.csv:      Ley de Hooke   F = k*x + F0        k=25.0 N/m,  F0=0.05 N
- decaimiento.csv:  Descarga RC    V = V0*exp(-t/tau)   V0=5.0 V,    tau=2.0 s
- pendulo.csv:      Pendulo simple T = a*L^b            a=2.0058 (=2*pi/sqrt(9.8)), b=0.5
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent
rng = np.random.default_rng(seed=42)


def make_resorte():
    k, F0 = 25.0, 0.05
    x = np.linspace(0.01, 0.20, 14)  # elongacion en metros
    F_true = k * x + F0
    F = F_true + rng.normal(0, 0.15, size=x.size)
    sigma_F = np.full_like(F, 0.15)
    df = pd.DataFrame({"x_m": x.round(4), "F_N": F.round(4), "sigma_F_N": sigma_F.round(4)})
    df.to_csv(OUT_DIR / "resorte.csv", index=False)


def make_decaimiento():
    V0, tau = 5.0, 2.0
    t = np.linspace(0, 8, 16)  # tiempo en segundos
    V_true = V0 * np.exp(-t / tau)
    V = V_true + rng.normal(0, 0.08, size=t.size)
    sigma_V = np.full_like(V, 0.08)
    df = pd.DataFrame({"t_s": t.round(3), "V_volt": V.round(4), "sigma_V_volt": sigma_V.round(4)})
    df.to_csv(OUT_DIR / "decaimiento.csv", index=False)


def make_pendulo():
    a, b = 2 * np.pi / np.sqrt(9.8), 0.5
    L = np.linspace(0.10, 1.20, 12)  # longitud en metros
    T_true = a * L ** b
    T = T_true + rng.normal(0, 0.03, size=L.size)
    sigma_T = np.full_like(T, 0.03)
    df = pd.DataFrame({"L_m": L.round(4), "T_s": T.round(4), "sigma_T_s": sigma_T.round(4)})
    df.to_csv(OUT_DIR / "pendulo.csv", index=False)


if __name__ == "__main__":
    make_resorte()
    make_decaimiento()
    make_pendulo()
    print(f"Datasets escritos en {OUT_DIR}/: resorte.csv, decaimiento.csv, pendulo.csv")
