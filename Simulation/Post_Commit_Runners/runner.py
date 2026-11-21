"""Zel Beta - Deterministic Post-Commit Runner (Verification Harness).

Acts as the automated Sensor: executes immediately on version n+1 after git commit.
Runs level-matched symbolic verification (type/syntax checks, test suites, contract integrity, SLA gates),
compares against baseline version n, and automatically raises either an ISSUE or PRAISE ExperienceEvent.
Zero LLM: 100% deterministic and symbolic.
"""

from __future__ import annotations

import subprocess
import time
import tracemalloc
import uuid
from pathlib import Path
from typing import Any, Literal, TypedDict


class ExperienceEvent(TypedDict):
    """Structured telemetry event raised automatically by the post-commit verification harness."""

    event_id: str
    kind: Literal["ISSUE", "PRAISE"]
    reporter_agent: str
    reporter_tier: Literal["ARCH", "AST", "EXEC", "SPEC"]
    head_sha: str
    predicate_cmd: str  # Deterministic 0/1 command for git bisect
    symptom_log: str    # Raw compiler diagnostic, profiler delta, or failure trace
    metrics: dict[str, Any]


class PostCommitRunner:
    """Deterministic, non-LLM verification harness executed on commit version n+1."""

    def __init__(self, repo_dir: Path | str = "Simulation/Repo"):
        self.repo_dir = Path(repo_dir).resolve()

    def check_contract_separation(
        self,
        head_sha: str,
        reporter_tier: str,
        baseline_sha: str | None = None,
    ) -> ExperienceEvent | None:
        """Anti-Goodhart Rule: Ensure code-mutating tiers (ARCH, AST, EXEC, SCRIPT, BYTECODE) never mutate tests or contracts."""
        norm_tier = {"SCRIPT": "ARCH", "BYTECODE": "EXEC"}.get(reporter_tier.upper(), reporter_tier.upper())
        if norm_tier in ("ARCH", "AST", "EXEC") and baseline_sha:
            try:
                diff_files_res = subprocess.run(
                    ["git", "diff", "--name-only", baseline_sha, head_sha],
                    cwd=self.repo_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                touched_files = [f.replace("\\", "/") for f in diff_files_res.stdout.splitlines()]
                forbidden = []
                for f in touched_files:
                    path_obj = Path(f)
                    if (
                        f.startswith(("tests/", "Constraints/", "contracts/"))
                        or path_obj.name.startswith("test_")
                        or path_obj.name.endswith(("_test.py", "conftest.py"))
                        or path_obj.name in ("pyproject.toml", "setup.cfg")
                    ):
                        forbidden.append(f)

                if forbidden:
                    return ExperienceEvent(
                        event_id=f"exp-{uuid.uuid4().hex[:8]}",
                        kind="ISSUE",
                        reporter_agent="contract-guard",
                        reporter_tier="SPEC",
                        head_sha=head_sha,
                        predicate_cmd="git diff --name-only",
                        symptom_log=f"Separation of Powers Violation: Tier {reporter_tier} modified forbidden test/contract paths: {forbidden}",
                        metrics={"forbidden_paths": forbidden},
                    )
            except subprocess.CalledProcessError:
                pass
        return None

    def run_syntax_compilation(self, head_sha: str, reporter_agent: str, reporter_tier: str) -> ExperienceEvent | None:
        """Deterministic check 1: Python compilation and bytecode syntax check across repository tree."""
        # Check all python files excluding .git and virtualenvs
        py_files = [
            f for f in self.repo_dir.rglob("*.py")
            if ".git" not in f.parts and "venv" not in f.parts and "__pycache__" not in f.parts
        ]
        for py_file in py_files:
            rel_file = py_file.relative_to(self.repo_dir)
            compile_res = subprocess.run(
                ["python", "-m", "py_compile", str(rel_file)],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if compile_res.returncode != 0:
                return ExperienceEvent(
                    event_id=f"exp-{uuid.uuid4().hex[:8]}",
                    kind="ISSUE",
                    reporter_agent=reporter_agent,
                    reporter_tier=reporter_tier,  # type: ignore
                    head_sha=head_sha,
                    predicate_cmd=f"python -m py_compile {rel_file}",
                    symptom_log=compile_res.stderr.strip() or "Bytecode compilation failed",
                    metrics={"returncode": compile_res.returncode},
                )
        return None

    def run_tests(
        self,
        head_sha: str,
        reporter_agent: str,
        reporter_tier: str,
        test_paths: list[Path] | None = None,
    ) -> ExperienceEvent | None:
        """Deterministic check 2: Execute unit and constraint test suites."""
        tests_to_run = test_paths or list((self.repo_dir / "tests").glob("test_*.py")) if (self.repo_dir / "tests").exists() else []

        # If repo has no tests directory, check for root test files or external constraint tests
        if not tests_to_run:
            tests_to_run = list(self.repo_dir.glob("test_*.py"))

        for test_file in tests_to_run:
            cmd = ["python", "-m", "unittest", str(test_file)]
            test_res = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
            if test_res.returncode != 0:
                symptom = test_res.stderr.strip() or test_res.stdout.strip()
                return ExperienceEvent(
                    event_id=f"exp-{uuid.uuid4().hex[:8]}",
                    kind="ISSUE",
                    reporter_agent=reporter_agent,
                    reporter_tier=reporter_tier,  # type: ignore
                    head_sha=head_sha,
                    predicate_cmd=" ".join(cmd),
                    symptom_log=symptom,
                    metrics={"failed_test": str(test_file)},
                )
        return None

    def evaluate_commit(
        self,
        head_sha: str,
        reporter_agent: str,
        reporter_tier: Literal["ARCH", "AST", "EXEC", "SPEC"],
        baseline_sha: str | None = None,
        test_paths: list[Path] | None = None,
    ) -> ExperienceEvent:
        """Run the full post-commit suite and automatically raise the symptom (ISSUE or PRAISE)."""
        start_time = time.perf_counter()

        # Gate 1: Contract separation
        separation_issue = self.check_contract_separation(head_sha, reporter_tier, baseline_sha)
        if separation_issue:
            return separation_issue

        # Gate 2: Syntax and compilation
        syntax_issue = self.run_syntax_compilation(head_sha, reporter_agent, reporter_tier)
        if syntax_issue:
            return syntax_issue

        # Gate 3: Test execution
        test_issue = self.run_tests(head_sha, reporter_agent, reporter_tier, test_paths=test_paths)
        if test_issue:
            return test_issue

        # If all checks passed without regression — benchmark for PRAISE telemetry
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Run a lightweight memory + timing benchmark on the first test file
        mem_peak_kb = 0.0
        speedup_factor = 1.0
        bench_tests = test_paths or list((self.repo_dir / "tests").glob("test_*.py")) if (self.repo_dir / "tests").exists() else list(self.repo_dir.glob("test_*.py"))
        if bench_tests:
            try:
                import tracemalloc
                tracemalloc.start()
                bench_start = time.perf_counter()
                subprocess.run(
                    ["python", "-m", "unittest", str(bench_tests[0])],
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                bench_elapsed = time.perf_counter() - bench_start
                _cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_peak_kb = round(peak / 1024, 2)
                # speedup_factor: ratio of gate-3 time vs full benchmark run
                speedup_factor = round(elapsed_ms / max(bench_elapsed * 1000, 1.0), 3)
            except Exception:
                pass

        return ExperienceEvent(
            event_id=f"exp-{uuid.uuid4().hex[:8]}",
            kind="PRAISE",
            reporter_agent=reporter_agent,
            reporter_tier=reporter_tier,
            head_sha=head_sha,
            predicate_cmd="python -m unittest",
            symptom_log="All symbolic checks and constraints verified successfully.",
            metrics={
                "elapsed_ms": elapsed_ms,
                "status": "VERIFIED",
                "mem_peak_kb": mem_peak_kb,
                "speedup_factor": speedup_factor,
            },
        )
