from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df[x], df[y], marker="o", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Год")
    plt.ylabel(ylabel)
    _savefig(out_path)


def plot_nominal_vs_ppp(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["year"], df["nominal_usd_rub"], marker="o", label="Номинальный USD/RUB")
    plt.plot(df["year"], df["ppp_basket_usd_rub"], marker="s", label="PPP корзинный курс")
    plt.title("Сравнение рыночного USD/RUB и курса по потребительской корзине")
    plt.xlabel("Год")
    plt.ylabel("RUB за 1 USD")
    plt.legend()
    _savefig(out_path)


def plot_scatter_levels(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[["log_brent", "log_basket_real_usd_rub"]].dropna()
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df["log_brent"], plot_df["log_basket_real_usd_rub"], alpha=0.8)
    plt.title("Scatter: log уровни")
    plt.xlabel("log_brent")
    plt.ylabel("log_basket_real_usd_rub")
    _savefig(out_path)


def plot_scatter_changes(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[["dlog_brent", "dlog_basket_real_usd_rub"]].dropna()
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df["dlog_brent"], plot_df["dlog_basket_real_usd_rub"], alpha=0.8)
    plt.title("Scatter: изменения логарифмов")
    plt.xlabel("dlog_brent")
    plt.ylabel("dlog_basket_real_usd_rub")
    _savefig(out_path)


def plot_residuals(residuals: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(residuals.index, residuals.values, marker="o", linewidth=1.4)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Остатки модели в изменениях (Модель 3)")
    plt.xlabel("Индекс наблюдения")
    plt.ylabel("Остатки")
    _savefig(out_path)
