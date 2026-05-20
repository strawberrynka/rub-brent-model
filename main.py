from __future__ import annotations

import logging
from pathlib import Path

from src.analysis import (
    run_adf_tests,
    run_cointegration_test,
    save_correlation_matrix,
    save_descriptive_statistics,
)
from src.data_loader import load_brent_fred, load_ppp_world_bank, load_usd_rub_fred
from src.modeling import (
    extract_residuals_for_plot,
    run_models,
    save_model_comparison,
    save_regression_results,
)
from src.plots import (
    plot_nominal_vs_ppp,
    plot_residuals,
    plot_scatter_changes,
    plot_scatter_levels,
    plot_time_series,
)
from src.preprocessing import build_final_dataset
from src.report_generator import generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    outputs_figures = project_root / "outputs" / "figures"
    outputs_tables = project_root / "outputs" / "tables"
    report_path = project_root / "outputs" / "report.md"

    start_date = "1992-01-01"
    end_date = None

    logger.info("Загрузка Brent из FRED...")
    brent_daily = load_brent_fred(start_date, end_date, data_raw)

    logger.info("Загрузка USD/RUB из FRED...")
    usd_rub_monthly = load_usd_rub_fred(start_date, end_date, data_raw)

    logger.info("Загрузка PPP (World Bank PA.NUS.PRVT.PP)...")
    ppp_yearly = load_ppp_world_bank(data_raw)

    logger.info("Предобработка и построение финального датасета...")
    df = build_final_dataset(brent_daily, usd_rub_monthly, ppp_yearly, data_processed)

    logger.info("Сохранение таблиц описательной статистики и корреляций...")
    descriptive_stats = save_descriptive_statistics(df, outputs_tables / "descriptive_statistics.csv")
    corr_matrix = save_correlation_matrix(df, outputs_tables / "correlation_matrix.csv")

    logger.info("Проведение тестов на стационарность и коинтеграцию...")
    adf_results = run_adf_tests(df, outputs_tables / "adf_tests.csv")
    coint_result = run_cointegration_test(df, outputs_tables / "cointegration_test.csv")

    logger.info("Оценка OLS-моделей...")
    models = run_models(df)
    regression_results = save_regression_results(models, outputs_tables / "regression_results.csv")
    model_comparison = save_model_comparison(models, outputs_tables / "model_comparison.csv")

    logger.info("Построение графиков...")
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

    residuals, _ = extract_residuals_for_plot(models)
    plot_residuals(residuals, outputs_figures / "residuals_model_changes.png")

    logger.info("Генерация отчета...")
    generate_report(
        df=df,
        descriptive_stats=descriptive_stats,
        corr_matrix=corr_matrix,
        regression_results=regression_results,
        model_comparison=model_comparison,
        models=models,
        adf_results=adf_results,
        coint_result=coint_result,
        output_path=report_path,
    )

    logger.info("Работа завершена успешно!")
    logger.info(f"Финальный датасет: {data_processed / 'final_dataset.csv'}")
    logger.info(f"Таблицы: {outputs_tables}")
    logger.info(f"Графики: {outputs_figures}")
    logger.info(f"Отчет: {report_path}")


if __name__ == "__main__":
    main()
