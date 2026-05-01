"""
Regression tests for dependency-graph edge resolution.

An earlier version resolved only relative imports, so absolute first-party
imports (`from core.models import User`) produced no edge at all. The graph then
had zero edges for most repositories and every ARCH_DETERMINISTIC layering or
cycle check passed vacuously, having traversed nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph_builder import build_graph  # noqa: E402

pytest.importorskip("networkx")


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_absolute_first_party_import_creates_edge(tmp_path: Path) -> None:
    _write(tmp_path, "core/__init__.py", "")
    _write(tmp_path, "web/__init__.py", "")
    _write(tmp_path, "core/models.py", "class User:\n    pass\n")
    _write(tmp_path, "web/views.py", "from core.models import User\n")

    graph = build_graph(str(tmp_path))

    assert ("web/views.py", "core/models.py") in graph.edges()


def test_relative_import_still_creates_edge(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/models.py", "class User:\n    pass\n")
    _write(tmp_path, "pkg/views.py", "from .models import User\n")

    graph = build_graph(str(tmp_path))

    assert ("pkg/views.py", "pkg/models.py") in graph.edges()


def test_dotted_import_binds_to_the_module_not_the_package(tmp_path: Path) -> None:
    _write(tmp_path, "core/__init__.py", "")
    _write(tmp_path, "core/models.py", "class User:\n    pass\n")
    _write(tmp_path, "app.py", "import core.models\n")

    graph = build_graph(str(tmp_path))

    assert ("app.py", "core/models.py") in graph.edges()


def test_third_party_and_stdlib_imports_produce_no_edge(tmp_path: Path) -> None:
    _write(tmp_path, "app.py", "import os\nimport networkx\nfrom json import loads\n")

    graph = build_graph(str(tmp_path))

    assert graph.number_of_edges() == 0


def test_import_cannot_escape_the_repo_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("TOKEN = 1\n", encoding="utf-8")
    repo = tmp_path / "repo"
    _write(repo, "app.py", "from outside.secret import TOKEN\n")

    graph = build_graph(str(repo))

    for _, target in graph.edges():
        assert not target.startswith("..")
        assert not Path(target).is_absolute()
