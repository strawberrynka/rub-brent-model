import hashlib
from pathlib import Path

from main import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_offline_build_is_reproducible_and_preserves_raw(tmp_path):
    raw = list((ROOT / "data" / "raw").glob("*.csv"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw}
    first = run_pipeline(ROOT, output_dir=tmp_path / "first")
    second = run_pipeline(ROOT, output_dir=tmp_path / "second", pdf=False)
    for path in (first["output_dir"] / "tables").glob("*.csv"):
        assert path.read_bytes() == (second["output_dir"] / "tables" / path.name).read_bytes()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in before.items())
    assert (first["output_dir"] / "report.pdf").read_bytes().startswith(b"%PDF")
    report = (first["output_dir"] / "report.md").read_text(encoding="utf-8")
    assert "1995-2025" in report
    assert "не причинный эффект" in report
    assert first["manifest"]["sources"] == second["manifest"]["sources"]
    assert first["manifest"]["code_sha256"] == second["manifest"]["code_sha256"]
