from __future__ import annotations

from pathlib import Path

from src.analysis import save_correlation_matrix, save_descriptive_statistics
from src.data_loader import load_brent_fred, load_ppp_world_bank, load_usd_rub_fred
from src.modeling import (
    extract_residuals_for_plot,
    run_models,
    save_model_comparison,
    save_regression_results,
    save_stationarity_tests,
)
from src.plots import (
    plot_nominal_vs_ppp,
    plot_residuals,
    plot_scatter_changes,
    plot_scatter_levels,
    plot_time_series,
)
from src.pdf_report_generator import build_pdf_report
from src.preprocessing import build_final_dataset
from src.report_generator import generate_report


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    outputs_figures = project_root / "outputs" / "figures"
    outputs_tables = project_root / "outputs" / "tables"
    report_path = project_root / "outputs" / "report.md"
    pdf_report_path = project_root / "outputs" / "report_detailed.pdf"

    start_date = "1992-01-01"
    end_date = None

    print("[INFO] Загрузка Brent из FRED...")
    brent_daily = load_brent_fred(start_date, end_date, data_raw)

    print("[INFO] Загрузка USD/RUB из FRED...")
    usd_rub_monthly = load_usd_rub_fred(start_date, end_date, data_raw)

    print("[INFO] Загрузка PPP (World Bank PA.NUS.PRVT.PP)...")
    ppp_yearly = load_ppp_world_bank(data_raw)

    print("[INFO] Предобработка и построение финального датасета...")
    df = build_final_dataset(brent_daily, usd_rub_monthly, ppp_yearly, data_processed)

    if len(df) < 15:
        print(
            f"[WARN] После объединения доступно только {len(df)} наблюдений. "
            "Результаты регрессий могут быть нестабильны."
        )

    print("[INFO] Сохранение таблиц описательной статистики и корреляций...")
    descriptive_stats = save_descriptive_statistics(df, outputs_tables / "descriptive_statistics.csv")
    corr_matrix = save_correlation_matrix(df, outputs_tables / "correlation_matrix.csv")

    print("[INFO] Оценка регрессионных моделей...")
    models = run_models(df)
    regression_results = save_regression_results(models, outputs_tables / "regression_results.csv")
    model_comparison = save_model_comparison(models, outputs_tables / "model_comparison.csv")
    stationarity_tests = save_stationarity_tests(df, outputs_tables / "stationarity_tests.csv")

    print("[INFO] Построение графиков...")
    plot_time_series(
        df,
        x="year",
        y="brent",
        title="Brent (годовые средние)",
        ylabel="USD за баррель",
        out_path=outputs_figures / "brent_time_series.png",
    )
    plot_time_series(
        df,
        x="year",
        y="nominal_usd_rub",
        title="Номинальный курс USD/RUB (годовые средние)",
        ylabel="RUB за 1 USD",
        out_path=outputs_figures / "nominal_usd_rub_time_series.png",
    )
    plot_time_series(
        df,
        x="year",
        y="ppp_basket_usd_rub",
        title="PPP курс USD/RUB по потребительской корзине",
        ylabel="RUB за международный доллар",
        out_path=outputs_figures / "ppp_basket_usd_rub_time_series.png",
    )
    plot_time_series(
        df,
        x="year",
        y="basket_real_usd_rub",
        title="Реальный корзинный курс доллара",
        ylabel="Индекс (nominal/ppp)",
        out_path=outputs_figures / "basket_real_usd_rub_time_series.png",
    )
    plot_nominal_vs_ppp(df, outputs_figures / "nominal_vs_ppp_rate.png")
    plot_scatter_levels(df, outputs_figures / "scatter_log_levels.png")
    plot_scatter_changes(df, outputs_figures / "scatter_log_changes.png")

    residuals, residual_model_name = extract_residuals_for_plot(models)
    plot_residuals(residuals, outputs_figures / "residuals_model_changes.png", residual_model_name)

    print("[INFO] Генерация отчета...")
    generate_report(
        df=df,
        descriptive_stats=descriptive_stats,
        corr_matrix=corr_matrix,
        regression_results=regression_results,
        model_comparison=model_comparison,
        stationarity_tests=stationarity_tests,
        models=models,
        output_path=report_path,
    )

    print("[INFO] Генерация PDF-отчета...")
    pdf_report_path = build_pdf_report(project_root)

    print("[DONE] Проект успешно выполнен.")
    print(f"[DONE] Финальный датасет: {data_processed / 'final_dataset.csv'}")
    print(f"[DONE] Таблицы: {outputs_tables}")
    print(f"[DONE] Графики: {outputs_figures}")
    print(f"[DONE] Отчет: {report_path}")
    print(f"[DONE] PDF-отчет: {pdf_report_path}")


if __name__ == "__main__":
    main()
