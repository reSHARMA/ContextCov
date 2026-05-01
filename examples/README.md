# Example

A self-contained demo you can run with no API key and no network.

`demo-repo/` is a four-file project with an `AGENTS.md` and three deliberate
violations of it. `demo.mapping.json` is the output of running `src.cli` against
that `AGENTS.md`.

## Run it

```bash
python -m src.scan --mapping examples/demo.mapping.json --root examples/demo-repo --no-semantic
```

```
Running source checks...
  Loaded 2 SOURCE_CHECK(s), scanned 4 file(s).
Build Failed (Source):
  core/models.py: Fail if any Python file contains tab characters used for indentation.
  web/views.py: Fail if any public function or method definition lacks a docstring.
Building dependency graph...
  Graph: 5 nodes, 1 edges.
Build Failed (Deterministic):
[Arch Violation] Files under web/ must not import directly from core/.
Violation: web/views.py imports core/models.py.
One or more check stages failed (see above).
```

Exit code 1. Each rule in `demo-repo/AGENTS.md` produced a check, and each check
found its planted violation:

| Rule in AGENTS.md | Check type | Caught |
|---|---|---|
| No tab indentation | `SOURCE_CHECK` | `core/models.py` uses a tab |
| Public functions need docstrings | `SOURCE_CHECK` | `web/views.py::view` has none |
| `web/` must not import `core/` | `ARCH_DETERMINISTIC` | `web/views.py` imports `core.models` |
| Never `git push --force` | `PROCESS_CHECK` | enforced at runtime, see below |

The architectural violation travels through an **absolute** import
(`from core.models import User`), which is resolved against the repository's
source roots when the import graph is built.

## Regenerate it

To rebuild the mapping from the instruction file (needs an LLM configured):

```bash
python -m src.cli raw/contextcov/demo/AGENTS.md \
  --local-readme examples/demo-repo/AGENTS.md \
  --repo-root examples/demo-repo \
  --output-mapping examples/demo.mapping.json \
  --force
```

## Try the runtime interception

The fourth rule cannot be checked by reading files — it has to stop a command:

```bash
cd examples/demo-repo
python -m src.setup_shims --mapping ../demo.mapping.json
PATH="$PWD/.contextcov/bin:$PATH" git push --force
```

## What is in a mapping

Each top-level key is a **stable ID**: a hash of the section's heading path and
its normalized text. That is what makes updates incremental — edit one section
and only that entry is regenerated.

```jsonc
{
  "9f2c…": {
    "stable_id": "9f2c…",
    "header_path": "Agent Instructions/Code style",
    "original_content": "Never use tab characters for indentation in Python files.",
    "compressed": "Use spaces instead of tab characters for indentation in Python files.",
    "strategies": [
      {
        "type": "SOURCE_CHECK",
        "confidence": 0.95,
        "trigger": "**/*.py",
        "directive": "Fail if any Python file contains tab characters used for indentation.",
        "static_check": {
          "target_lang": "text",
          "code": "result = None\nfor line in source_text.splitlines():\n    ..."
        }
      }
    ]
  }
}
```

- `original_content` is the rule as you wrote it; `compressed` is the single
  constraint it was rewritten into.
- `trigger` selects which files the check runs against.
- `directive` is what gets reported when the check fails.
- `code` is plain Python, executed with `tree`, `source_bytes`, `source_text` and
  `re` in scope. It is meant to be read and edited — if a generated check is
  wrong, fix it in place.
