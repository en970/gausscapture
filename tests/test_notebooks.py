"""The notebooks are the only documented GPU path, so their imports are checked.

The 4D notebook's pre-flight cell imports the training entry point and calls
``raise SystemExit('fix the import above before training')`` when that fails.
It named ``gausscapture.recon.deform.train4d.train_4d``, which does not exist --
``train4d`` defines ``train``; ``train_4d`` lives in ``gausscapture.recon.fit4d``
-- so the only documented way to train a scene aborted at cell 3, before a single
iteration, and 417 passing tests said nothing about it because nothing in
``tests/`` had ever opened a notebook.

A notebook cannot be executed here (no GPU, no Drive, no Colab runtime), but its
*import lines* are ordinary Python and can be. They are also the part that rots:
a module rename in ``src/`` is invisible to a JSON file. So every
``from gausscapture... import ...`` in every notebook -- including the ones
written inside a string handed to ``subprocess`` -- is extracted and executed.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))

#: Matches the import whether it is a bare source line or embedded in a quoted
#: string handed to a subprocess, which is how the pre-flight cell writes it.
_IMPORT = re.compile(r"from\s+(gausscapture[\w.]*)\s+import\s+([\w,\s*]+)")


def gausscapture_imports() -> list[tuple[str, str, str]]:
    """``(notebook name, module, symbol)`` for every import a notebook performs."""
    found: list[tuple[str, str, str]] = []
    for notebook in NOTEBOOKS:
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            for line in cell.get("source", []):
                for module, names in _IMPORT.findall(line):
                    for name in names.replace("\\n", " ").split(","):
                        symbol = name.strip().strip("'\"")
                        if symbol and symbol != "*":
                            found.append((notebook.name, module, symbol))
    return found


def test_there_are_notebooks_to_check():
    """A silent empty parametrisation is the failure mode this file exists for."""
    assert NOTEBOOKS, "no notebooks found; this test would otherwise pass vacuously"
    assert gausscapture_imports(), "no gausscapture imports found in any notebook"


@pytest.mark.parametrize(
    ("notebook", "module", "symbol"),
    gausscapture_imports(),
    ids=lambda value: str(value),
)
def test_every_notebook_import_resolves(notebook: str, module: str, symbol: str):
    """No GPU needed: the trainer's own dependency check happens inside the call."""
    imported = importlib.import_module(module)
    assert hasattr(imported, symbol), (
        f"{notebook} imports {symbol} from {module}, which does not define it. "
        "The notebook is the only documented GPU path; it fails at the pre-flight cell."
    )


def test_the_4d_notebook_trains_through_the_cli_it_documents():
    """The training cell shells out to `gausscapture.cli train4d`; that must exist."""
    from gausscapture.cli import _build_parser

    text = (Path(__file__).resolve().parents[1] / "notebooks" / "GaussCapture_Colab_4D.ipynb").read_text(
        encoding="utf-8"
    )
    assert "'train4d'" in text
    # argparse raises SystemExit on an unknown subcommand, which is the whole check.
    parsed = _build_parser().parse_args(
        ["train4d", "dataset4d.zip", "--out", "scene4d.npz", "--device", "cuda"]
    )
    assert parsed.device == "cuda"
