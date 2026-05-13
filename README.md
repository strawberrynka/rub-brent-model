# Модель связи реального курса доллара по потребительской корзине и цены нефти Brent

- Бойко София, БПМИ248
- Шумакова Екатерина, БПМИ2310

## 1. Краткое описание проекта
Проект оценивает связь между ценой нефти Brent и реальным курсом доллара к рублю, рассчитанным через паритет покупательной способности для частного потребления (PPP private consumption), а не через CPI-индексы.

## 2. Экономический смысл корзинного реального курса
Используется показатель World Bank `PA.NUS.PRVT.PP` для России как курс USD/RUB по потребительской корзине:

- `ppp_basket_usd_rub` — сколько рублей нужно в России для сопоставимого частного потребления, которое 1 доллар обеспечивает в США.
- `nominal_usd_rub` — рыночный курс RUB за 1 USD.
- `basket_real_usd_rub = nominal_usd_rub / ppp_basket_usd_rub`.

Интерпретация:
- `basket_real_usd_rub > 1`: рыночный доллар дороже PPP-ориентира;
- `basket_real_usd_rub < 1`: рыночный доллар дешевле PPP-ориентира.

## 3. Источники данных
- Brent oil price: FRED `DCOILBRENTEU`
- Nominal USD/RUB: FRED `CCUSMA02RUM618N`
- PPP private consumption (Russia): World Bank `PA.NUS.PRVT.PP`
  - API: `https://api.worldbank.org/v2/country/RUS/indicator/PA.NUS.PRVT.PP?format=json&per_page=20000`

## 4. Инструкция по запуску
1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Запустить полный pipeline:

```bash
python main.py
```

## 5. Структура проекта

```text
rub-brent-model/
  README.md
  requirements.txt
  main.py
  src/
    data_loader.py
    preprocessing.py
    analysis.py
    modeling.py
    plots.py
  data/
    raw/
    processed/
  outputs/
    figures/
    tables/
    report.md [Был перемещен в родительскую папку rub-brent-model]
```

## 6. Выходные файлы
- Итоговый датасет: `data/processed/final_dataset.csv`
- Графики: `outputs/figures/*.png`
- Таблицы:
  - `outputs/tables/descriptive_statistics.csv`
  - `outputs/tables/correlation_matrix.csv`
  - `outputs/tables/regression_results.csv`
  - `outputs/tables/model_comparison.csv`
- Отчет: `report.md`
