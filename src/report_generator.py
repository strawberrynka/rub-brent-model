"""One report content model shared by Markdown and PDF."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import Config


@dataclass
class Section:
    title: str
    paragraphs: list[str]
    tables: list[pd.DataFrame] = field(default_factory=list)
    image: str | None = None


def fmt(value: float) -> str:
    return f"{value:.4f}" if abs(value) >= 0.0001 else f"{value:.2e}"


def build_sections(df, coverage, tables: dict, config: Config, manifest: dict) -> list[Section]:
    comparison = tables["model_comparison"]
    reg = tables["regression_results"]
    ref = config.model.reference_model
    effect = reg.loc[reg.model.eq(ref) & reg.parameter.eq("dlog_brent")].iloc[0]
    baseline = comparison.set_index("model").loc["ar1"]
    candidate = comparison.set_index("model").loc["ar1_oil"]
    best_aic = comparison.loc[comparison.aic.idxmin(), "model"]
    best_bic = comparison.loc[comparison.bic.idxmin(), "model"]
    excluded = coverage.loc[~coverage.included, ["year", "exclusion_reason"]].rename(
        columns={"year": "Год", "exclusion_reason": "Причина исключения"}
    )
    excluded["Причина исключения"] = (
        excluded["Причина исключения"]
        .str.replace("incomplete_fx", "Неполный год курса", regex=False)
        .str.replace("incomplete_brent", "Недостаточное покрытие Brent", regex=False)
        .str.replace("missing_ppp", "Нет ППС", regex=False)
    )
    if excluded.empty:
        excluded = pd.DataFrame({"Проверка": ["Все запрошенные годы прошли контроль"]})
    precision = (
        "Интервал не включает ноль."
        if effect.ci_low * effect.ci_high > 0
        else "Интервал включает ноль; оценка совместима с отсутствием связи."
    )
    summary = [
        "Вопрос: добавляет ли изменение цены Brent информацию об изменении отношения рыночного "
        "курса USD/RUB к ППС потребления сверх собственной инерции этого показателя?",
        f"Полные годы: {int(df.year.min())}-{int(df.year.max())}; уровни: {len(df)} наблюдения. "
        f"Общая выборка моделей изменений: {int(candidate.start_year)}-{int(candidate.end_year)}, "
        f"N = {int(candidate.n_obs)}.",
        f"В основной спецификации {ref} коэффициент Brent = {fmt(effect.coef)}; "
        f"95% HAC-интервал [{fmt(effect.ci_low)}; {fmt(effect.ci_high)}], "
        f"p = {fmt(effect.hac_p_value)}. {precision} Это условная связь, не причинный эффект.",
        f"Добавление текущего Brent к AR(1): изменение AIC = {candidate.aic - baseline.aic:.3f}, "
        f"BIC = {candidate.bic - baseline.bic:.3f} (отрицательное значение означает улучшение). "
        "Это качество описания внутри выборки, не проверка прогноза.",
    ]
    compare_view = comparison[["model", "n_obs", "adj_r_squared", "aic", "bic", "bg_f_p_value"]].rename(
        columns={
            "model": "Модель",
            "n_obs": "N",
            "adj_r_squared": "Adj. R²",
            "aic": "AIC",
            "bic": "BIC",
            "bg_f_p_value": "BG p",
        }
    )
    influence = tables["influence"].nlargest(5, "cooks_distance")[["year", "cooks_distance", "leverage"]]
    influence = influence.rename(
        columns={"year": "Год", "cooks_distance": "Cook D", "leverage": "Leverage"}
    )
    loo = tables["leave_one_out"]
    station = tables["stationarity_tests"].copy()
    station["KPSS p"] = station.apply(lambda r: str(r.kpss_p_bound) + fmt(r.kpss_p_value), axis=1)
    station = station[["variable", "adf_p_value", "KPSS p"]].rename(
        columns={"variable": "Ряд", "adf_p_value": "ADF p"}
    )
    cointegration = tables["cointegration_test"]
    coint_view = cointegration[["y0", "maxlag", "coint_t", "p_value"]].rename(
        columns={"y0": "Зависимый ряд", "maxlag": "Лаг", "coint_t": "Статистика", "p_value": "p"}
    )
    primary = cointegration.loc[cointegration.primary].iloc[0]
    primary_text = (
        "нулевая гипотеза отсутствия коинтеграции отвергается на уровне 5%"
        if primary.p_value < 0.05
        else "нет достаточных оснований отвергнуть отсутствие коинтеграции на уровне 5%"
    )
    run_info = f"Режим данных: {manifest['data']['mode']}. Дата исходной загрузки: "
    run_info += manifest["data"].get("fetched_at_utc") or "не зафиксирована в историческом снимке"
    return [
        Section("Нефть и рубль: результаты", summary, [excluded], "overview.png"),
        Section(
            "Сравнение спецификаций",
            [
                "Все модели в таблице объясняют Δlog(q) на одинаковых годах. ar1 - лаг изменения q; "
                "ar1_oil добавляет текущее изменение Brent; ar1_oil_lag - также лаг Brent; "
                "ar1_oil_crises - кризисные индикаторы. changes и changes_lag не включают лаг q.",
                f"Минимум AIC: {best_aic}; минимум BIC: {best_bic}. Основная спецификация {ref} "
                "зафиксирована в конфигурации, а не выбрана по минимальному p-value. "
                "Модель уровней сохранена отдельно и не участвует в этом рейтинге.",
                f"HAC, число лагов: {config.model.hac_lags}; поправка малой выборки: "
                f"{'включена' if config.model.hac_small_sample_correction else 'выключена'}; t-интервалы. "
                f"BG - F-версия теста Breusch-Godfrey с {config.model.diagnostic_lags} лагом; "
                "неотклонение гипотезы не доказывает отсутствия автокорреляции.",
            ],
            [compare_view],
            "brent_coefficients.png",
        ),
        Section(
            "Устойчивость и влиятельные годы",
            [
                "Начало выборки и число лагов HAC меняются по заранее заданной сетке config.toml. "
                "При смене начала выборки разности и лаги пересчитываются заново; "
                "постоянные кризисные индикаторы исключаются с записью в таблицу результатов.",
                f"При исключении по одному наблюдению из фиксированной матрицы основной модели "
                f"коэффициент Brent меняется от {loo.brent_coef.min():.4f} до {loo.brent_coef.max():.4f}. "
                "Это диагностическая проверка влияния; её нельзя считать временной кросс-валидацией. "
                "Ниже - пять наибольших значений расстояния Кука; наблюдения автоматически не удаляются.",
            ],
            [influence],
            "sensitivity.png",
        ),
        Section(
            "Стационарность и долгосрочная связь",
            [
                f"ADF: константа и фиксированные {config.model.adf_maxlag} лага. Нулевая гипотеза ADF - "
                "единичный корень; KPSS - стационарность около константы, число лагов выбирается автоматически. "
                "Знаки >= и <= обозначают границы табличных p-value KPSS. "
                "При малой выборке и структурных разрывах мощность тестов ограничена.",
                f"Основной Engle-Granger: log(q) на log(Brent), константа; число лагов "
                f"фиксировано: {config.model.coint_maxlag}. p = {fmt(primary.p_value)}: "
                f"{primary_text}. Таблица показывает чувствительность к направлению регрессии и лагам. "
                "Интерпретация теста предполагает I(1) для обоих рядов; здесь это не установленный факт.",
            ],
            [station, coint_view],
        ),
        Section(
            "Методика, источники и воспроизведение",
            [
                "q = E / PPP, где E - RUB за USD, PPP - RUB за международный доллар по потреблению. "
                "Рост q соответствует относительному ослаблению рубля. ППС - статистическая оценка, "
                "а отклонение q от единицы само по себе не доказывает рыночной недооценки.",
                f"В годе требуется 12 месяцев курса, 12 месяцев Brent и не менее "
                f"{config.data.min_brent_observations} наблюдений Brent. "
                "Для курса используется равновзвешенное среднее месячных средних; "
                "для нефти - среднее доступных дневных цен. Это различие весов сохраняется явно. "
                "Пропуски внутри итогового календаря останавливают расчёт.",
                "Ограничения: малая годовая выборка, пересмотры и задержка публикации ППС, "
                "неучтённые факторы, смена валютных режимов и влияние ранних лет. "
                "Кризисные индикаторы учитывают выбранные периоды, но не выделяют причинный нефтяной канал. "
                "Brent текущего года неизвестен в начале этого года; модель не является готовым прогнозом.",
                run_info
                + ". Контрольные суммы источников, конфигурация, версии библиотек и состояние Git "
                "сохранены в run_manifest.json. Markdown и PDF строятся из одного набора результатов.",
                "Повторение: python main.py --offline. Обновление источников: python main.py --refresh. "
                "Проверки: python -m pytest. Параметры: config.toml.",
                "Авторы исходного исследования: Бойко София и Шумакова Екатерина. "
                "Сведения о распределении исходной работы приведены в README.",
                "Источники: FRED DCOILBRENTEU (https://fred.stlouisfed.org/series/DCOILBRENTEU); "
                "FRED CCUSMA02RUM618N (https://fred.stlouisfed.org/series/CCUSMA02RUM618N); "
                "World Bank PA.NUS.PRVT.PP (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP).",
                "Методические справки: https://www.statsmodels.org/stable/generated/"
                "statsmodels.tsa.stattools.coint.html ; "
                "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/AIC.html",
            ],
        ),
    ]


def generate_report(sections: list[Section], output_path: Path) -> None:
    lines = [
        "# Нефть и рубль: устойчивость связи",
        "",
        "Автоматически сформировано из результатов расчёта.",
        "",
    ]
    for section in sections:
        lines += [f"## {section.title}", "", *[p + "\n" for p in section.paragraphs]]
        for table in section.tables:
            lines += [table.to_markdown(index=False, floatfmt=".4f"), ""]
        if section.image:
            lines += [f"![{section.title}](figures/{section.image})", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
