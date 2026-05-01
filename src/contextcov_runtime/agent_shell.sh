#!/usr/bin/env bash
# Export PATH with .contextcov/bin first, then exec bash so tools spawned via $SHELL inherit shims.
_contextcov_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PATH="${_contextcov_root}/bin:${PATH}"
exec bash "$@"
