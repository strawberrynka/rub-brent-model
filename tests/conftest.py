from pathlib import Path
import pytest
import requests

from src.config import load_config
from src.data_loader import load_sources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests and offline builds must not access the network")

    monkeypatch.setattr(requests, "get", blocked)


@pytest.fixture
def sources():
    return load_sources(ROOT / "data" / "raw")[0]


@pytest.fixture
def config():
    return load_config(ROOT / "config.toml")
