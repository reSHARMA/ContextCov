"""
Integration tests for the ContextCov shim system.

Tests the full pipeline: shimmed command -> dispatcher.sh -> check_process.py -> real binary
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.setup_shims import build_compliance_db_from_mapping, setup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test isolation."""
    td = tempfile.mkdtemp(prefix="contextcov_test_")
    yield Path(td)
    shutil.rmtree(td, ignore_errors=True)


@pytest.fixture
def mock_binary(temp_dir: Path):
    """
    Create a mock binary that prints its name and args to stdout.
    Returns a function to create mock binaries with a given name.
    """
    bin_dir = temp_dir / "mock_bin"
    bin_dir.mkdir()

    def create(name: str) -> Path:
        script = bin_dir / name
        script.write_text(
            f"""#!/bin/bash
echo "REAL_BINARY_RAN: {name}"
echo "ARGS: $@"
exit 0
"""
        )
        script.chmod(0o755)
        return script

    return create


@pytest.fixture
def contextcov_dir(temp_dir: Path):
    """Create a .contextcov directory structure for testing."""
    cc_dir = temp_dir / ".contextcov"
    cc_dir.mkdir()
    (cc_dir / "bin").mkdir()
    (cc_dir / "runtime").mkdir()
    return cc_dir


# ---------------------------------------------------------------------------
# Unit tests for build_compliance_db_from_mapping
# ---------------------------------------------------------------------------


class TestBuildComplianceDB:
    def test_empty_mapping(self):
        """Empty mapping produces empty DB."""
        assert build_compliance_db_from_mapping({}) == {}

    def test_extracts_process_checks(self):
        """Extracts PROCESS_CHECK strategies with code."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm install",
                        "process_check": {
                            "code": "result = (True, '')",
                        },
                    }
                ]
            }
        }
        db = build_compliance_db_from_mapping(mapping)
        assert len(db) == 1
        entry = list(db.values())[0]
        assert entry["type"] == "PROCESS"
        assert entry["trigger"] == "npm"  # Normalized to first word
        assert entry["code"] == "result = (True, '')"

    def test_ignores_non_process_checks(self):
        """Ignores strategies that are not PROCESS_CHECK."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "SOURCE_CHECK",
                        "trigger": "npm",
                        "source_check": {"code": "pass"},
                    }
                ]
            }
        }
        db = build_compliance_db_from_mapping(mapping)
        assert db == {}

    def test_ignores_process_checks_without_code(self):
        """Ignores PROCESS_CHECK without code."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm",
                        "process_check": {},
                    }
                ]
            }
        }
        db = build_compliance_db_from_mapping(mapping)
        assert db == {}

    def test_priority_and_enforcement_level(self):
        """Extracts priority and enforcement_level if present."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm",
                        "process_check": {
                            "code": "result = (True, '')",
                            "priority": 10,
                            "enforcement_level": "block",
                        },
                    }
                ]
            }
        }
        db = build_compliance_db_from_mapping(mapping)
        entry = list(db.values())[0]
        assert entry["priority"] == 10
        assert entry["enforcement_level"] == "block"


# ---------------------------------------------------------------------------
# Unit tests for check_process.py logic
# ---------------------------------------------------------------------------


class TestCheckProcessLogic:
    """Test the check execution logic (same as what check_process.py does)."""

    def run_check(self, code: str, args: list, env: dict | None = None, trigger: str = "test") -> tuple[bool, str]:
        """Run a check in the same way check_process.py does."""
        env = env or {}
        code_context = {
            "trigger": trigger,
            "args": args,
            "env": env,
            "cwd": os.getcwd(),
            "os": os,
            "sys": sys,
            "result": (True, ""),
        }
        try:
            exec(code, code_context)
            passed, message = code_context.get("result", (True, ""))
            return bool(passed), message
        except Exception as e:
            return False, f"[Runtime error] {e}"

    def test_allow_when_result_true(self):
        """Check allows when result is (True, ...)."""
        code = "result = (True, 'allowed')"
        passed, msg = self.run_check(code, [])
        assert passed is True

    def test_block_when_result_false(self):
        """Check blocks when result is (False, ...)."""
        code = "result = (False, 'blocked by test')"
        passed, msg = self.run_check(code, [])
        assert passed is False
        assert "blocked by test" in msg

    def test_args_available_in_check(self):
        """Check has access to args."""
        code = """
if '--force' in args:
    result = (False, 'force flag not allowed')
else:
    result = (True, '')
"""
        passed, _ = self.run_check(code, ["--force"])
        assert passed is False

        passed, _ = self.run_check(code, ["--safe"])
        assert passed is True

    def test_env_available_in_check(self):
        """Check has access to environment variables."""
        code = """
if env.get('CI') == 'true':
    result = (True, 'CI allowed')
else:
    result = (False, 'must run in CI')
"""
        passed, _ = self.run_check(code, [], env={"CI": "true"})
        assert passed is True

        passed, _ = self.run_check(code, [], env={})
        assert passed is False

    def test_runtime_error_returns_false(self):
        """Runtime errors in check code return (False, error message)."""
        code = "raise ValueError('intentional error')"
        passed, msg = self.run_check(code, [])
        assert passed is False
        assert "Runtime error" in msg


# ---------------------------------------------------------------------------
# Integration tests for the full shim pipeline
# ---------------------------------------------------------------------------


class TestShimPipeline:
    """
    Integration tests for the full pipeline:
    shimmed command -> dispatcher.sh -> check_process.py -> real binary
    """

    def setup_shim_environment(
        self, temp_dir: Path, trigger: str, check_code: str, mock_binary_path: Path
    ) -> tuple[Path, dict]:
        """
        Set up a complete shim environment.
        Returns (contextcov_dir, env_dict_for_subprocess).
        """
        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        # Write compliance_db.json
        db = {
            "test_check_0": {
                "type": "PROCESS",
                "trigger": trigger,
                "code": check_code,
            }
        }
        (cc_dir / "compliance_db.json").write_text(json.dumps(db))

        # Copy runtime scripts from src
        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh", "agent_shell.sh"):
            src = src_runtime / name
            dst = runtime_dir / name
            shutil.copy2(src, dst)
            if name.endswith(".sh"):
                dst.chmod(0o755)

        # Create symlink for the trigger command
        dispatcher = runtime_dir / "dispatcher.sh"
        shim = bin_dir / trigger
        shim.symlink_to(dispatcher.resolve())

        # Build PATH: shim dir first, then mock binary dir
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{mock_binary_path.parent}:{env.get('PATH', '')}"

        return cc_dir, env

    def test_shim_allows_and_runs_real_binary(self, temp_dir: Path, mock_binary):
        """When check passes, shim should run the real binary."""
        # Create mock binary
        real_bin = mock_binary("testcmd")

        # Set up shim that always allows
        check_code = "result = (True, 'allowed')"
        cc_dir, env = self.setup_shim_environment(temp_dir, "testcmd", check_code, real_bin)

        # Run the shimmed command
        result = subprocess.run(
            ["testcmd", "arg1", "arg2"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert result.returncode == 0
        assert "REAL_BINARY_RAN: testcmd" in result.stdout
        assert "ARGS: arg1 arg2" in result.stdout

    def test_shim_blocks_when_check_fails(self, temp_dir: Path, mock_binary):
        """When check fails, shim should block and not run the real binary."""
        real_bin = mock_binary("blockedcmd")

        # Set up shim that always blocks
        check_code = "result = (False, 'Command blocked by policy')"
        cc_dir, env = self.setup_shim_environment(temp_dir, "blockedcmd", check_code, real_bin)

        result = subprocess.run(
            ["blockedcmd", "arg1"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert result.returncode != 0
        assert "REAL_BINARY_RAN" not in result.stdout
        assert "ContextCov Violation" in result.stderr
        assert "Command blocked by policy" in result.stderr

    def test_shim_passes_args_to_check(self, temp_dir: Path, mock_binary):
        """Shim should pass arguments to the check for inspection."""
        real_bin = mock_binary("argcmd")

        # Block if --dangerous flag is present
        check_code = """
if '--dangerous' in args:
    result = (False, 'dangerous flag not allowed')
else:
    result = (True, '')
"""
        cc_dir, env = self.setup_shim_environment(temp_dir, "argcmd", check_code, real_bin)

        # Should be blocked
        result = subprocess.run(
            ["argcmd", "--dangerous"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )
        assert result.returncode != 0
        assert "dangerous flag not allowed" in result.stderr

        # Should be allowed
        result = subprocess.run(
            ["argcmd", "--safe"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )
        assert result.returncode == 0
        assert "REAL_BINARY_RAN: argcmd" in result.stdout

    def test_shim_with_no_db_allows_command(self, temp_dir: Path, mock_binary):
        """When compliance_db.json is missing, command should be allowed."""
        real_bin = mock_binary("nocheckcmd")

        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        # Copy runtime scripts but don't create compliance_db.json
        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh"):
            src = src_runtime / name
            dst = runtime_dir / name
            shutil.copy2(src, dst)
            if name.endswith(".sh"):
                dst.chmod(0o755)

        # Create symlink
        dispatcher = runtime_dir / "dispatcher.sh"
        shim = bin_dir / "nocheckcmd"
        shim.symlink_to(dispatcher.resolve())

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{real_bin.parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["nocheckcmd"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert result.returncode == 0
        assert "REAL_BINARY_RAN: nocheckcmd" in result.stdout


# ---------------------------------------------------------------------------
# Tests for setup_shims.py setup() function
# ---------------------------------------------------------------------------


class TestSetupShims:
    """Test the setup() function that creates .contextcov from a mapping file."""

    def test_setup_creates_directory_structure(self, temp_dir: Path):
        """setup() creates .contextcov with bin/, runtime/, and compliance_db.json."""
        # Create a mapping file
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm",
                        "process_check": {"code": "result = (True, '')"},
                    }
                ]
            }
        }
        mapping_file = temp_dir / "test.mapping.json"
        mapping_file.write_text(json.dumps(mapping))

        cc_dir = temp_dir / ".contextcov"
        setup(mapping_file, cc_dir)

        assert cc_dir.exists()
        assert (cc_dir / "bin").exists()
        assert (cc_dir / "runtime").exists()
        assert (cc_dir / "compliance_db.json").exists()
        assert (cc_dir / "runtime" / "dispatcher.sh").exists()
        assert (cc_dir / "runtime" / "check_process.py").exists()
        assert (cc_dir / "runtime" / "agent_shell.sh").exists()

    def test_setup_creates_symlinks_for_triggers(self, temp_dir: Path):
        """setup() creates symlinks in bin/ for each trigger."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm install",
                        "process_check": {"code": "result = (True, '')"},
                    },
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "yarn add",
                        "process_check": {"code": "result = (True, '')"},
                    },
                ]
            }
        }
        mapping_file = temp_dir / "test.mapping.json"
        mapping_file.write_text(json.dumps(mapping))

        cc_dir = temp_dir / ".contextcov"
        setup(mapping_file, cc_dir)

        bin_dir = cc_dir / "bin"
        assert (bin_dir / "npm").is_symlink()
        assert (bin_dir / "yarn").is_symlink()

    def test_setup_dispatcher_is_executable(self, temp_dir: Path):
        """dispatcher.sh should be executable after setup."""
        mapping = {
            "rule_1": {
                "strategies": [
                    {
                        "type": "PROCESS_CHECK",
                        "trigger": "npm",
                        "process_check": {"code": "result = (True, '')"},
                    }
                ]
            }
        }
        mapping_file = temp_dir / "test.mapping.json"
        mapping_file.write_text(json.dumps(mapping))

        cc_dir = temp_dir / ".contextcov"
        setup(mapping_file, cc_dir)

        dispatcher = cc_dir / "runtime" / "dispatcher.sh"
        assert os.access(dispatcher, os.X_OK)


# ---------------------------------------------------------------------------
# Tests for agent_shell.sh
# ---------------------------------------------------------------------------


class TestAgentShell:
    """Test the agent_shell.sh wrapper."""

    def test_agent_shell_prepends_bin_to_path(self, temp_dir: Path, mock_binary):
        """agent_shell.sh should prepend .contextcov/bin to PATH."""
        # Set up a simple shim
        real_bin = mock_binary("testshell")
        check_code = "result = (True, '')"

        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        # Write compliance_db.json
        db = {"test_0": {"type": "PROCESS", "trigger": "testshell", "code": check_code}}
        (cc_dir / "compliance_db.json").write_text(json.dumps(db))

        # Copy runtime scripts
        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh", "agent_shell.sh"):
            shutil.copy2(src_runtime / name, runtime_dir / name)
            if name.endswith(".sh"):
                (runtime_dir / name).chmod(0o755)

        # Create shim symlink
        (bin_dir / "testshell").symlink_to((runtime_dir / "dispatcher.sh").resolve())

        # Run a command through agent_shell.sh
        agent_shell = runtime_dir / "agent_shell.sh"
        env = os.environ.copy()
        env["PATH"] = f"{real_bin.parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            [str(agent_shell), "-c", "testshell hello"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert result.returncode == 0
        assert "REAL_BINARY_RAN: testshell" in result.stdout


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_check_with_syntax_error_fails_open(self, temp_dir: Path, mock_binary):
        """A check with syntax error should fail open (allow command)."""
        real_bin = mock_binary("syntaxerr")

        # Invalid Python syntax
        check_code = "result = (True, ''  # missing closing paren"

        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        db = {"test_0": {"type": "PROCESS", "trigger": "syntaxerr", "code": check_code}}
        (cc_dir / "compliance_db.json").write_text(json.dumps(db))

        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh"):
            shutil.copy2(src_runtime / name, runtime_dir / name)
            if name.endswith(".sh"):
                (runtime_dir / name).chmod(0o755)

        (bin_dir / "syntaxerr").symlink_to((runtime_dir / "dispatcher.sh").resolve())

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{real_bin.parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["syntaxerr"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        # Should fail open and run the real binary
        assert result.returncode == 0
        assert "REAL_BINARY_RAN: syntaxerr" in result.stdout
        assert "Internal Error" in result.stderr

    def test_multiple_checks_for_same_trigger(self, temp_dir: Path, mock_binary):
        """Multiple checks for the same trigger should all run; first block wins."""
        real_bin = mock_binary("multicmd")

        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        # Two checks: first allows, second blocks
        db = {
            "check_1": {
                "type": "PROCESS",
                "trigger": "multicmd",
                "code": "result = (True, 'check 1 allows')",
                "priority": 10,
            },
            "check_2": {
                "type": "PROCESS",
                "trigger": "multicmd",
                "code": "result = (False, 'check 2 blocks')",
                "priority": 5,
            },
        }
        (cc_dir / "compliance_db.json").write_text(json.dumps(db))

        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh"):
            shutil.copy2(src_runtime / name, runtime_dir / name)
            if name.endswith(".sh"):
                (runtime_dir / name).chmod(0o755)

        (bin_dir / "multicmd").symlink_to((runtime_dir / "dispatcher.sh").resolve())

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{real_bin.parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["multicmd"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        # Higher priority check runs first and allows, then lower priority blocks
        assert result.returncode != 0
        assert "check 2 blocks" in result.stderr

    def test_command_with_spaces_in_args(self, temp_dir: Path, mock_binary):
        """Arguments with spaces should be passed correctly."""
        real_bin = mock_binary("spacecmd")
        check_code = "result = (True, '')"

        cc_dir = temp_dir / ".contextcov"
        bin_dir = cc_dir / "bin"
        runtime_dir = cc_dir / "runtime"
        bin_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)

        db = {"test_0": {"type": "PROCESS", "trigger": "spacecmd", "code": check_code}}
        (cc_dir / "compliance_db.json").write_text(json.dumps(db))

        src_runtime = _PROJECT_ROOT / "src" / "contextcov_runtime"
        for name in ("check_process.py", "dispatcher.sh"):
            shutil.copy2(src_runtime / name, runtime_dir / name)
            if name.endswith(".sh"):
                (runtime_dir / name).chmod(0o755)

        (bin_dir / "spacecmd").symlink_to((runtime_dir / "dispatcher.sh").resolve())

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{real_bin.parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["spacecmd", "arg with spaces", "another arg"],
            env=env,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert result.returncode == 0
        assert "ARGS: arg with spaces another arg" in result.stdout
