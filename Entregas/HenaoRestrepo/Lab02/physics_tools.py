"""
Herramientas (tools) que el agente puede usar para ajustar datos
experimentales de fisica. Cada funcion tiene tipos y un docstring corto:
la libreria `ollama` los usa para generar el schema que el LLM ve.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "resorte": {
        "file": "resorte.csv",
        "x_col": "x_m",
        "y_col": "F_N",
        "sigma_col": "sigma_F_N",
        "description": "Ley de Hooke: elongacion x (m) vs fuerza F (N) de un resorte.",
    },
    "decaimiento": {
        "file": "decaimiento.csv",
        "x_col": "t_s",
        "y_col": "V_volt",
        "sigma_col": "sigma_V_volt",
        "description": "Descarga de un circuito RC: tiempo t (s) vs voltaje V (V).",
    },
    "pendulo": {
        "file": "pendulo.csv",
        "x_col": "L_m",
        "y_col": "T_s",
        "sigma_col": "sigma_T_s",
        "description": "Pendulo simple: longitud L (m) vs periodo de oscilacion T (s).",
    },
}


def _modelo_lineal(x, a, b):
    return a * x + b


def _modelo_exponencial(x, a, b, c):
    return a * np.exp(b * x) + c


def _modelo_potencia(x, a, b):
    return a * np.power(x, b)


MODELS = {
    "lineal": {"func": _modelo_lineal, "formula": "y = a*x + b", "p0": [1.0, 0.0]},
    "exponencial": {"func": _modelo_exponencial, "formula": "y = a*exp(b*x) + c", "p0": [1.0, -0.5, 0.0]},
    "potencia": {"func": _modelo_potencia, "formula": "y = a * x^b", "p0": [1.0, 0.5]},
}


def _load(dataset: str) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ValueError(f"Dataset desconocido: {dataset}. Usa list_datasets() para ver los disponibles.")
    return pd.read_csv(DATA_DIR / DATASETS[dataset]["file"])


def list_datasets() -> str:
    """Lista los datasets experimentales disponibles y una breve descripcion de cada uno."""
    return json.dumps(
        {name: meta["description"] for name, meta in DATASETS.items()},
        ensure_ascii=False,
    )


def preview_dataset(dataset: str, n: int = 5) -> str:
    """Muestra las primeras n filas de un dataset experimental, para inspeccionar sus columnas y valores."""
    df = _load(dataset)
    return df.head(n).to_json(orient="records")


def list_models() -> str:
    """Lista las familias de modelos matematicos disponibles para ajustar curvas (lineal, exponencial, potencia) con su formula."""
    return json.dumps({name: meta["formula"] for name, meta in MODELS.items()}, ensure_ascii=False)


def fit_dataset(dataset: str, model: str) -> str:
    """Ajusta un modelo (lineal, exponencial o potencia) a un dataset experimental usando minimos cuadrados y devuelve los parametros ajustados, su incertidumbre y el R^2."""
    if model not in MODELS:
        raise ValueError(f"Modelo desconocido: {model}. Usa list_models() para ver los disponibles.")

    df = _load(dataset)
    meta = DATASETS[dataset]
    x = df[meta["x_col"]].to_numpy()
    y = df[meta["y_col"]].to_numpy()
    sigma = df[meta["sigma_col"]].to_numpy() if meta["sigma_col"] in df.columns else None

    spec = MODELS[model]
    popt, pcov = curve_fit(spec["func"], x, y, p0=spec["p0"], sigma=sigma, absolute_sigma=sigma is not None)
    perr = np.sqrt(np.diag(pcov))

    y_pred = spec["func"](x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    param_names = ["a", "b", "c"][: len(popt)]
    result = {
        "dataset": dataset,
        "model": model,
        "formula": spec["formula"],
        "params": {name: round(float(v), 5) for name, v in zip(param_names, popt)},
        "param_uncertainty": {name: round(float(v), 5) for name, v in zip(param_names, perr)},
        "r_squared": round(float(r2), 5),
    }
    return json.dumps(result, ensure_ascii=False)


def plot_dataset_fit(dataset: str, model: str) -> str:
    """Genera y guarda una imagen PNG con los puntos experimentales, la curva ajustada y los residuos; devuelve la ruta del archivo guardado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model not in MODELS:
        raise ValueError(f"Modelo desconocido: {model}. Usa list_models() para ver los disponibles.")

    df = _load(dataset)
    meta = DATASETS[dataset]
    x = df[meta["x_col"]].to_numpy()
    y = df[meta["y_col"]].to_numpy()
    sigma = df[meta["sigma_col"]].to_numpy() if meta["sigma_col"] in df.columns else None

    spec = MODELS[model]
    popt, _ = curve_fit(spec["func"], x, y, p0=spec["p0"], sigma=sigma, absolute_sigma=sigma is not None)

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = spec["func"](x_fit, *popt)
    residuals = y - spec["func"](x, *popt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True)
    ax1.errorbar(x, y, yerr=sigma, fmt="o", label="datos", capsize=3)
    ax1.plot(x_fit, y_fit, "-", label=f"ajuste ({model})")
    ax1.set_ylabel(meta["y_col"])
    ax1.set_title(f"{dataset}: {meta['description']}")
    ax1.legend()

    ax2.axhline(0, color="gray", lw=1)
    ax2.errorbar(x, residuals, yerr=sigma, fmt="o", capsize=3)
    ax2.set_xlabel(meta["x_col"])
    ax2.set_ylabel("residuos")

    fig.tight_layout()
    out_path = OUT_DIR / f"{dataset}_{model}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return json.dumps({"plot_path": str(out_path)}, ensure_ascii=False)
