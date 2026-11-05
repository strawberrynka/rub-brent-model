from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ModelConfig

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "text.color": "#172033",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
BLUE, TEAL = "#2855A6", "#008477"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    df: pd.DataFrame,
    regression: pd.DataFrame,
    sensitivity: pd.DataFrame,
    reference,
    out_dir: Path,
    config: ModelConfig,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(df.year, df.brent, color=BLUE, linewidth=2)
    axes[0].set(ylabel="USD за баррель", title="Brent и отношение рыночного курса к ППС")
    axes[1].plot(df.year, df.basket_real_usd_rub, color=TEAL, linewidth=2)
    axes[1].set(ylabel="q = USD/RUB ÷ ППС", xlabel="Год")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    _save(fig, out_dir / "overview.png")

    effects = regression.loc[
        (regression.parameter == "dlog_brent") & regression.group.eq("changes")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    y = np.arange(len(effects))
    ax.errorbar(
        effects.coef,
        y,
        xerr=[effects.coef - effects.ci_low, effects.ci_high - effects.coef],
        fmt="o",
        color=BLUE,
        capsize=4,
    )
    ax.set_yticks(y, effects.label)
    ax.axvline(0, color="#94A3B8", linestyle="--")
    ax.set(
        xlabel="Коэффициент при изменении Brent; 95% HAC-интервал",
        title="Зависимость оценки от спецификации",
    )
    ax.grid(axis="x", alpha=0.2)
    _save(fig, out_dir / "brent_coefficients.png")

    selected = sensitivity.loc[
        sensitivity.status.eq("ok") & sensitivity.hac_lags.eq(config.hac_lags)
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    y = np.arange(len(selected))
    labels = [f"{row.model} | с {int(row.start_year_requested)}" for row in selected.itertuples()]
    ax.errorbar(
        selected.coef,
        y,
        xerr=[selected.coef - selected.ci_low, selected.ci_high - selected.coef],
        fmt="o",
        color=TEAL,
        capsize=4,
    )
    ax.set_yticks(y, labels, fontsize=9)
    ax.axvline(0, color="#94A3B8", linestyle="--")
    ax.set(xlabel="Коэффициент Brent; 95% HAC-интервал", title="Устойчивость к началу выборки")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, out_dir / "sensitivity.png")

    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    ax.plot(reference.result.resid.index, reference.result.resid, color=BLUE, marker="o", markersize=4)
    ax.axhline(0, color="#94A3B8", linestyle="--")
    ax.set(xlabel="Год", ylabel="Остаток, Δlog(q)", title=f"Остатки: {reference.name}")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out_dir / "residuals.png")
