"""Run the complete analysis on the checked-in snapshot by default."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rub-brent-matplotlib"))

from src.analysis import save_correlation_matrix, save_descriptive_statistics
from src.config import load_config
from src.data_loader import SOURCES, load_sources
from src.modeling import (
    cointegration_table,
    comparison_table,
    influence_tables,
    regression_table,
    run_models,
    save_table,
    sensitivity_table,
    stationarity_table,
)
from src.pdf_report_generator import build_pdf_report
from src.plots import create_figures
from src.preprocessing import build_final_dataset
from src.report_generator import build_sections, generate_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, raw_dir: Path, config, source_info: dict) -> dict:
    source_files = [root / "main.py", *sorted((root / "src").glob("*.py"))]
    code_hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in source_files}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty_at_run": dirty,
        "code_sha256": code_hashes,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in [
                "pandas",
                "numpy",
                "scipy",
                "statsmodels",
                "matplotlib",
                "reportlab",
                "requests",
            ]
        },
        "config": config.to_dict(),
        "data": source_info,
        "sources": {
            name: {**info, "sha256": _sha256(raw_dir / info["file"])} for name, info in SOURCES.items()
        },
    }


def run_pipeline(
    project_root: Path,
    *,
    config_path: Path | None = None,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    refresh: bool = False,
    pdf: bool = True,
) -> dict:
    root = project_root.resolve()
    config = load_config(config_path or root / "config.toml")
    raw_dir = (raw_dir or root / "data" / "raw").resolve()
    output = (output_dir or root / "outputs").resolve()
    processed = root / "data" / "processed" if output == root / "outputs" else output / "data"
    output.mkdir(parents=True, exist_ok=True)
    figures, tables_dir = output / "figures", output / "tables"
    frames, source_info = load_sources(raw_dir, refresh=refresh)
    df, coverage = build_final_dataset(
        frames["brent"], frames["usd_rub"], frames["ppp"], processed, config.data
    )
    models = run_models(df, config.model)
    ref = config.model.reference_model
    if ref not in {"ar1_oil", "ar1_oil_lag", "ar1_oil_crises"}:
        raise ValueError("reference_model must be a dynamic model with Brent")
    influence, loo = influence_tables(models[ref])
    tables = {
        "model_comparison": comparison_table(models, config.model),
        "regression_results": regression_table(models, config.model),
        "stationarity_tests": stationarity_table(df, config.model),
        "cointegration_test": cointegration_table(df, config.model),
        "sensitivity": sensitivity_table(df, config.model),
        "influence": influence,
        "leave_one_out": loo,
        "data_coverage": coverage,
    }
    levels = models["levels"].result
    import pandas as pd

    tables["levels_model"] = pd.DataFrame(
        [
            {
                "model": "levels",
                "n_obs": int(levels.nobs),
                "start_year": int(df.year.min()),
                "end_year": int(df.year.max()),
                "r_squared": levels.rsquared,
                "aic": levels.aic,
                "bic": levels.bic,
                "note": "Exploratory only; not comparable to change models",
            }
        ]
    )
    for name, frame in tables.items():
        save_table(frame, tables_dir / f"{name}.csv")
    save_descriptive_statistics(df, tables_dir / "descriptive_statistics.csv")
    save_correlation_matrix(df, tables_dir / "correlation_matrix.csv")
    create_figures(
        df, tables["regression_results"], tables["sensitivity"], models[ref], figures, config.model
    )
    manifest = _manifest(root, raw_dir, config, source_info)
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sections = build_sections(df, coverage, tables, config, manifest)
    generate_report(sections, output / "report.md")
    if pdf:
        build_pdf_report(sections, output / "report.pdf", figures)
    print(
        f"Complete: {len(df)} full years; "
        f"{int(tables['model_comparison'].iloc[0].n_obs)} common regression observations."
    )
    print(f"Report: {output / 'report.md'}")
    return {
        "data": df,
        "coverage": coverage,
        "models": models,
        "tables": tables,
        "manifest": manifest,
        "config": config,
        "output_dir": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true", help="Use checked-in snapshot (default)")
    modes.add_argument("--refresh", action="store_true", help="Fetch and replace raw sources explicitly")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        Path(__file__).resolve().parent,
        config_path=args.config,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        refresh=args.refresh,
        pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
