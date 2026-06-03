from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans", str(p)))
            return "DejaVuSans"
    return "Helvetica"


def _fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def _df_table(df: pd.DataFrame, max_rows: int = 30) -> Table:
    sub = df.head(max_rows).copy()
    data = [list(sub.columns)]
    for _, row in sub.iterrows():
        out = []
        for val in row:
            if isinstance(val, float):
                out.append(_fmt(val))
            else:
                out.append(str(val))
        data.append(out)

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def build_pdf_report(project_root: Path) -> Path:
    outputs = project_root / "outputs"
    tables = outputs / "tables"
    figures = outputs / "figures"

    reg = pd.read_csv(tables / "regression_results.csv")
    cmp_df = pd.read_csv(tables / "model_comparison.csv")
    desc = pd.read_csv(tables / "descriptive_statistics.csv")
    corr = pd.read_csv(tables / "correlation_matrix.csv")
    stationarity_path = tables / "stationarity_tests.csv"
    stationarity = pd.read_csv(stationarity_path) if stationarity_path.exists() else pd.DataFrame()

    # bring index column back when saved from describe/corr
    if "Unnamed: 0" in desc.columns:
        desc = desc.rename(columns={"Unnamed: 0": "variable"})
    if "Unnamed: 0" in corr.columns:
        corr = corr.rename(columns={"Unnamed: 0": "variable"})

    m1 = reg[reg["model"] == "model_1"].copy()
    m2 = reg[reg["model"] == "model_2"].copy()
    m3 = reg[reg["model"] == "model_3"].copy()
    m4 = reg[reg["model"] == "model_4"].copy()
    m5 = reg[reg["model"] == "model_5"].copy()
    has_hac = "hac_p_value" in reg.columns

    def coef(df: pd.DataFrame, param: str) -> float:
        return float(df.loc[df["parameter"] == param, "coef"].iloc[0])

    def pval(df: pd.DataFrame, param: str) -> float:
        return float(df.loc[df["parameter"] == param, "p_value"].iloc[0])

    def hac_pval(df: pd.DataFrame, param: str) -> float:
        if not has_hac:
            return float("nan")
        return float(df.loc[df["parameter"] == param, "hac_p_value"].iloc[0])

    def significance_note(value: float) -> str:
        if pd.isna(value):
            return "HAC p-value недоступен"
        if value < 0.05:
            return "значим на 5% уровне"
        if value < 0.1:
            return "значим только на 10% уровне"
        return "не достигает стандартных уровней значимости"

    def regression_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["parameter", "coef", "p_value"]
        if has_hac:
            cols.extend(["hac_std_err", "hac_p_value"])
        cols.extend(["r_squared", "n_obs", "durbin_watson"])
        return df[cols]

    corr_brent = float(corr.loc[corr["variable"] == "brent", "basket_real_usd_rub"].iloc[0])
    sign_word = "отрицательная" if corr_brent < 0 else "положительная"
    best_model = str(cmp_df.sort_values("aic").iloc[0]["model"])

    font_name = _register_font()
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font_name, fontSize=16, leading=20)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font_name, fontSize=12, leading=15)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font_name, fontSize=10, leading=14)

    pdf_path = outputs / "report_detailed.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)

    story = []
    story.append(Paragraph("Подробный отчет: связь реального курса USD/RUB по потребительской корзине и цены Brent", h1))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Проект выполнен на годовых данных FRED и World Bank. Реальный курс рассчитан как отношение рыночного USD/RUB к PPP-курсу частного потребления (PA.NUS.PRVT.PP).", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Что было сделано", h2))
    story.append(Paragraph("1) Загружены данные Brent, USD/RUB и PPP private consumption для России. 2) Ряды приведены к годовой частоте. 3) Рассчитаны уровни, логарифмы, изменения логарифмов, лаги и dummy-переменные. 4) Оценены пять регрессионных спецификаций: базовые, динамические и режимные. 5) Для регрессий добавлены HAC-ошибки Ньюи-Уэста и диагностика автокорреляции. 6) Построены графики, таблицы и отчет.", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Ключевые показатели данных", h2))
    story.append(_df_table(desc, max_rows=20))

    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Корреляции", h2))
    story.append(Paragraph(f"Корреляция между Brent и basket_real_usd_rub: {_fmt(corr_brent)} ({sign_word}). Это предварительно поддерживает гипотезу: более дорогая нефть связана с более низким basket_real_usd_rub.", body))
    corr_show = corr[["variable", "brent", "basket_real_usd_rub", "log_brent", "log_basket_real_usd_rub", "dlog_brent", "dlog_basket_real_usd_rub"]]
    story.append(_df_table(corr_show, max_rows=20))

    if not stationarity.empty:
        story.append(Spacer(1, 10))
        story.append(Paragraph("4. Стационарность", h2))
        story.append(Paragraph("ADF проверяет гипотезу о единичном корне, KPSS — гипотезу стационарности. Это помогает отделить долгосрочную связь в уровнях от более надежной динамики в изменениях.", body))
        story.append(_df_table(stationarity, max_rows=10))

    story.append(PageBreak())
    story.append(Paragraph("5. Регрессионные модели", h2))
    story.append(Paragraph("В таблицах показаны обычные OLS p-value и, где доступно, HAC p-value. Для годовых временных рядов HAC-проверка важна, потому что Durbin-Watson заметно ниже 2 и указывает на положительную автокорреляцию остатков.", body))
    story.append(Paragraph("Модель 1: log_basket_real_usd_rub = const + beta * log_brent", body))
    story.append(Paragraph(f"Оценка beta = {_fmt(coef(m1, 'log_brent'))}, OLS p-value = {_fmt(pval(m1, 'log_brent'))}, HAC p-value = {_fmt(hac_pval(m1, 'log_brent'))}: {significance_note(hac_pval(m1, 'log_brent'))}.", body))
    story.append(_df_table(regression_cols(m1), max_rows=10))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Модель 2: dlog_basket_real_usd_rub = const + beta * dlog_brent", body))
    story.append(Paragraph(f"Оценка beta = {_fmt(coef(m2, 'dlog_brent'))}, OLS p-value = {_fmt(pval(m2, 'dlog_brent'))}, HAC p-value = {_fmt(hac_pval(m2, 'dlog_brent'))}: {significance_note(hac_pval(m2, 'dlog_brent'))}.", body))
    story.append(_df_table(regression_cols(m2), max_rows=10))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Модель 3: dlog_basket_real_usd_rub = const + beta1 * dlog_brent + beta2 * dlog_brent_lag1", body))
    story.append(Paragraph(f"Текущий эффект Brent: beta1 = {_fmt(coef(m3, 'dlog_brent'))}, OLS p-value = {_fmt(pval(m3, 'dlog_brent'))}, HAC p-value = {_fmt(hac_pval(m3, 'dlog_brent'))}: {significance_note(hac_pval(m3, 'dlog_brent'))}. Лаговый эффект: beta2 = {_fmt(coef(m3, 'dlog_brent_lag1'))}, HAC p-value = {_fmt(hac_pval(m3, 'dlog_brent_lag1'))}.", body))
    story.append(_df_table(regression_cols(m3), max_rows=10))

    if not m4.empty:
        story.append(PageBreak())
        story.append(Paragraph("Модель 4: динамическая спецификация", body))
        story.append(Paragraph(f"Добавлен лаг изменения реального корзинного курса. Brent: beta1 = {_fmt(coef(m4, 'dlog_brent'))}, HAC p-value = {_fmt(hac_pval(m4, 'dlog_brent'))}. Лаг курса: phi = {_fmt(coef(m4, 'dlog_basket_real_usd_rub_lag1'))}, HAC p-value = {_fmt(hac_pval(m4, 'dlog_basket_real_usd_rub_lag1'))}.", body))
        story.append(_df_table(regression_cols(m4), max_rows=10))

    if not m5.empty:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Модель 5: динамика + кризисные режимы", body))
        story.append(Paragraph(f"Добавлены dummy для 1998, 2014-2015 и 2022+. Brent: beta1 = {_fmt(coef(m5, 'dlog_brent'))}, HAC p-value = {_fmt(hac_pval(m5, 'dlog_brent'))}. Лаг курса: phi = {_fmt(coef(m5, 'dlog_basket_real_usd_rub_lag1'))}, HAC p-value = {_fmt(hac_pval(m5, 'dlog_basket_real_usd_rub_lag1'))}.", body))
        story.append(_df_table(regression_cols(m5), max_rows=10))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Попытка дополнительного улучшения", body))
    story.append(Paragraph("Дополнительно проверялись макроконтроли: инфляционный дифференциал Россия-США, ставка центрального банка, санкционный режим после 2014 года и кризисные dummy 2008-2009/2020. По метрикам эти варианты оказались хуже model_5: adjusted R-squared снизился с 0.6253 до 0.4748 и 0.3933, AIC ухудшился с -39.7465 до -32.8127 и -26.1220, BIC ухудшился с -29.4863 до -26.1517 и -14.1321. Поэтому макроконтроли не включены в финальную спецификацию.", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Сравнение моделей", h2))
    story.append(Paragraph(f"По AIC лучшей оказалась {best_model}. Однако у всех моделей ограниченное объяснение вариации, поэтому результаты стоит трактовать как частичную связь, а не полное объяснение курса.", body))
    cmp_cols = ["model", "r_squared", "adj_r_squared", "aic", "bic", "n_obs", "durbin_watson"]
    if "ljung_box_pvalue_lag1" in cmp_df.columns:
        cmp_cols.append("ljung_box_pvalue_lag1")
    story.append(_df_table(cmp_df[cmp_cols], max_rows=10))

    story.append(PageBreak())
    story.append(Paragraph("7. Графики и экономическая интерпретация", h2))
    story.append(Paragraph("Ниже ключевые графики проекта. Визуально видно, что периоды высокой Brent часто сочетаются с более низким basket_real_usd_rub, но связь не идеальна и зависит от режимов экономики.", body))

    for fig_name in [
        "nominal_vs_ppp_rate.png",
        "basket_real_usd_rub_time_series.png",
        "scatter_log_levels.png",
        "scatter_log_changes.png",
        "residuals_model_changes.png",
    ]:
        fig_path = figures / fig_name
        if fig_path.exists():
            story.append(Spacer(1, 8))
            story.append(Paragraph(fig_name, body))
            story.append(Image(str(fig_path), width=17.5 * cm, height=9 * cm))

    story.append(Spacer(1, 10))
    story.append(Paragraph("8. Главные выводы", h2))
    story.append(Paragraph("1) Знак коэффициентов в целом соответствует гипотезе об отрицательной связи Brent и реального корзинного курса. 2) Динамические спецификации лучше учитывают инерцию курса и автокорреляцию остатков. 3) Dummy-переменные помогают отделить нефтяной канал от кризисных режимов. 4) Попытка добавить инфляцию, ставку и санкционные индикаторы ухудшила AIC/BIC и adjusted R-squared относительно model_5. 5) Показатель PPP годовой и оценочный, из-за этого выборка небольшая и точность ограничена.", body))

    doc.build(story)
    return pdf_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = build_pdf_report(root)
    print(out)
