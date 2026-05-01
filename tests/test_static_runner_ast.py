"""
Regression tests for AST-check behaviour when the tree-sitter backend is unusable.

An incompatible tree-sitter / tree-sitter-languages pair imports cleanly but
raises when constructing a Language. The runner used to swallow that and execute
AST checks with ``tree=None``, so every such check reported a pass it had never
evaluated. These tests pin the contract in both directions: with a working
backend the check runs; without one it is skipped and reported, never passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import static_runner  # noqa: E402
from src.static_runner import (  # noqa: E402
    run_source_checks_for_repo,
    tree_sitter_backend_ok,
)

AST_CHECK_MAPPING = {
    "seg_1": {
        "strategies": [
            {
                "type": "SOURCE_CHECK",
                "trigger": "**/*.py",
                "directive": "flag every function definition",
                "static_check": {
                    "target_lang": "python",
                    "code": (
                        "result = None\n"
                        "for node in tree.root_node.children:\n"
                        "    if node.type == 'function_definition':\n"
                        "        result = 'FAIL'\n"
                    ),
                },
            }
        ]
    }
}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def test_ast_check_runs_when_backend_is_available(repo: Path) -> None:
    if not tree_sitter_backend_ok():
        pytest.skip("tree-sitter backend unavailable in this environment")

    violations, stats = run_source_checks_for_repo(repo, AST_CHECK_MAPPING, return_stats=True)

    assert stats["ast_backend_ok"] is True
    assert stats["checks_skipped_no_parser"] == 0
    assert len(violations) == 1


def test_ast_check_is_skipped_not_passed_when_backend_is_broken(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead parser must never turn an AST check into a silent pass."""
    monkeypatch.setattr(static_runner, "_get_parser", lambda lang: None)

    violations, stats = run_source_checks_for_repo(repo, AST_CHECK_MAPPING, return_stats=True)

    assert stats["ast_backend_ok"] is False
    # The check must be accounted for as skipped rather than silently passing.
    assert stats["checks_skipped_no_parser"] >= 1
    assert violations == []


def test_text_checks_are_unaffected_by_a_broken_backend(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(static_runner, "_get_parser", lambda lang: None)
    mapping = {
        "seg_1": {
            "strategies": [
                {
                    "type": "SOURCE_CHECK",
                    "trigger": "**/*.py",
                    "directive": "no def keyword",
                    "static_check": {
                        "target_lang": "text",
                        "code": "result = 'FAIL' if 'def ' in source_text else None\n",
                    },
                }
            ]
        }
    }

    violations, stats = run_source_checks_for_repo(repo, mapping, return_stats=True)

    assert stats["checks_skipped_no_parser"] == 0
    assert len(violations) == 1
