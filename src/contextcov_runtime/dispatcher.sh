#!/bin/bash
# ContextCov Shim Dispatcher: intercepts commands, runs compliance checks, then execs real binary.
# Symlinked as e.g. .contextcov/bin/npm so when PATH has .contextcov/bin first, "npm" runs this.
#
# This script must not invoke ANY command resolved through PATH before it has
# sanitized PATH, because PATH starts with our own shim directory. A shim for
# bash, env, basename, dirname or python3 would otherwise re-enter this script
# without bound. Accordingly:
#   - the shebang is an absolute path (setup_shims rewrites it to the bash it found)
#   - $0 is split with shell parameter expansion, not basename/dirname
#   - every external command is resolved through CLEAN_PATH (shim dir removed)

set -e

# 0. Abort loudly if we have somehow re-entered ourselves.
CONTEXTCOV_DEPTH=$((${CONTEXTCOV_DEPTH:-0} + 1))
export CONTEXTCOV_DEPTH
if [ "$CONTEXTCOV_DEPTH" -gt 10 ]; then
  echo "ContextCov: dispatcher re-entered $CONTEXTCOV_DEPTH times; aborting to avoid an exec loop." >&2
  echo "  A PROCESS_CHECK trigger probably shims a command the dispatcher itself needs." >&2
  exit 70
fi

# 1. Who we are (e.g. "npm") - parameter expansion, no basename.
CMD_NAME="${0##*/}"

# 2. Resolve .contextcov root (parent of directory containing this script).
# When symlinked from .contextcov/bin/npm, $0 is .../bin/npm, so strip the last
# path segment to get .../bin, then go up one.
case "$0" in
  */*) SCRIPT_DIR="$(cd "${0%/*}" && pwd)" ;;
  *)   SCRIPT_DIR="$(pwd)" ;;
esac
CONTEXTCOV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME="${CONTEXTCOV_ROOT}/runtime"
CHECK_SCRIPT="${RUNTIME}/check_process.py"

# 3. Build a PATH with our shim directory removed, once, and use it for every
# external command below.
CLEAN_PATH=""
_old_ifs="$IFS"
IFS=':'
for _dir in $PATH; do
  case "$_dir" in
    "$CONTEXTCOV_ROOT"*) continue ;;
  esac
  if [ -z "$CLEAN_PATH" ]; then
    CLEAN_PATH="$_dir"
  else
    CLEAN_PATH="${CLEAN_PATH}:${_dir}"
  fi
done
IFS="$_old_ifs"

# 4. Find a real binary using CLEAN_PATH only.
_find_real_binary() {
  cmd="$1"
  _old_ifs="$IFS"
  IFS=':'
  for dir in $CLEAN_PATH; do
    if [ -x "$dir/$cmd" ]; then
      IFS="$_old_ifs"
      echo "$dir/$cmd"
      return 0
    fi
  done
  IFS="$_old_ifs"
  return 1
}

# 5. Resolve the real python3 for running the compliance check.
# Use `if ! VAR=$(...)` so `set -e` does not abort before the error message.
if ! REAL_PYTHON3=$(_find_real_binary python3); then
  echo "ContextCov: real python3 not found in PATH (excluding shim dir)" >&2
  exit 127
fi

# 6. Run compliance check
if [ ! -f "$CHECK_SCRIPT" ]; then
  echo "ContextCov: check_process.py not found at ${CHECK_SCRIPT}" >&2
  exit 127
fi
CHECK_EXIT_CODE=0
"$REAL_PYTHON3" "$CHECK_SCRIPT" "$CMD_NAME" "$@" || CHECK_EXIT_CODE=$?

# 7. If check failed, stop
if [ "$CHECK_EXIT_CODE" -ne 0 ]; then
  exit "$CHECK_EXIT_CODE"
fi

if ! REAL_BINARY=$(_find_real_binary "$CMD_NAME"); then
  echo "ContextCov Error: Could not find real '$CMD_NAME' binary in PATH." >&2
  echo "  The shim at $0 passed compliance checks, but the actual '$CMD_NAME' is not installed or not in PATH." >&2
  echo "  PATH (excluding ContextCov shims): $CLEAN_PATH" >&2
  exit 127
fi

# 8. Execute real binary with original arguments
exec "$REAL_BINARY" "$@"
