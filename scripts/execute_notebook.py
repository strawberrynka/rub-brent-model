"""Execute the research notebook using this interpreter, without a global kernel install."""

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    runtime = build / "jupyter-runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    notebook = nbformat.read(ROOT / "notebooks" / "research.ipynb", as_version=4)
    with tempfile.TemporaryDirectory(dir=build, prefix="kernel-") as directory:
        kernel = Path(directory) / "rub-brent"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "rub-brent",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="rub-brent", kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory])
        )
        NotebookClient(
            notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}
        ).execute()
    target = build / "research.executed.ipynb"
    nbformat.write(notebook, target)
    print(f"Executed notebook: {target}")


if __name__ == "__main__":
    main()
