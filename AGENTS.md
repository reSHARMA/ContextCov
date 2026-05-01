# Agent Instructions

Instructions for coding agents working in the ContextCov repository.

ContextCov compiles this file into executable checks, so these rules are also a
worked example of what the tool can enforce. Generate them with:

```bash
python -m src.cli raw/reSHARMA/ContextCov/AGENTS.md --local-readme ./AGENTS.md --repo-root .
```

## Code style

- Every public function must have a docstring explaining what it does.
- Never use tab characters for indentation in Python files.
- Prefer explicit exception types over bare `except:` clauses.

## Error handling

- Never swallow an exception silently. If an error is caught and not re-raised,
  the handler must log or report it — a bare `except Exception: pass` hides real
  failures from users.
- A check that cannot be evaluated must be reported as skipped, never as passed.
  Silently degrading to a pass tells the user their constraint holds when it was
  never inspected.

## Architecture

- Modules under `src/` must not import from the repository root except for `llm`
  and `llm_cache`.
- The shim dispatcher (`src/contextcov_runtime/dispatcher.sh`) must not invoke
  any command resolved through `PATH` before it has sanitized `PATH`. It runs
  with the shim directory first on `PATH`, so a shimmed `bash` or `python3` would
  otherwise make it re-enter itself without bound.

## Testing

- Run `pytest tests/` before committing.
- A bug fix must come with a regression test that fails without the fix. Both
  the dead Tree-sitter parser and the edgeless import graph shipped because the
  existing suite passed identically with them broken.

## Workflow

- Never run `git push --force` on this repository.
- Never commit a `.env` file or any file containing credentials.
