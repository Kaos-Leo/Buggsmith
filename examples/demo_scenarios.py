"""Zel Beta - Curated Demo Scenario Pack.

Provides three pre-built, reproducible software engineering scenarios:
- Scenario A (data_pipeline.py): AST & EXEC optimization convergence (Loops -> ListComps, Slots, Constant Folding)
- Scenario B (math_kernel.py): Multi-hunk regression localization via git bisect + ddmin delta-debugging
- Scenario C (contract_breach.py): Anti-Goodhart Separation of Powers enforcement (tamper defense)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import ingest
from Simulation.Constraints.generator import ConstraintManager
from Simulation.Grader import FeedbackPacket, HardForensicGrader
from Simulation.Post_Commit_Runners.runner import ExperienceEvent, PostCommitRunner
from Simulation.simulation import SimulationEngine
from Simulation.Transformer.coordinator import TransformerCoordinator


DATA_PIPELINE_CODE = '''class Record:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class DataPipeline:
    def __init__(self):
        self.records = []
        self.valid_types = ["A", "B", "C", "D"]

    def process(self, items):
        result = []
        for item in items:
            result.append(item * 2)
        return result

    def filter_valid(self, records):
        output = []
        for r in records:
            if r.name in ["A", "B", "C", "D"]:
                output.append(r)
        return output

    def compute_stats(self):
        BATCH_SIZE = 32 * 1024
        TIMEOUT = 60 * 60 * 24
        return {"batch": BATCH_SIZE, "timeout": TIMEOUT}
'''

DATA_PIPELINE_TEST = '''import unittest
from data_pipeline import Record, DataPipeline

class TestDataPipeline(unittest.TestCase):
    def test_process(self):
        pipe = DataPipeline()
        self.assertEqual(pipe.process([1, 2, 3]), [2, 4, 6])

    def test_filter(self):
        pipe = DataPipeline()
        r1 = Record("A", 10)
        r2 = Record("Z", 99)
        filtered = pipe.filter_valid([r1, r2])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "A")

    def test_stats(self):
        pipe = DataPipeline()
        stats = pipe.compute_stats()
        self.assertEqual(stats["batch"], 32768)
        self.assertEqual(stats["timeout"], 86400)

if __name__ == "__main__":
    unittest.main()
'''

MATH_KERNEL_BASE_CODE = '''import math

def normalize(values):
    total = sum(values)
    if total == 0:
        return [0.0] * len(values)
    return [v / total for v in values]

def dot_product(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(x * y)
    return sum(result)

def compute_scale():
    return 60 * 60 * 24
'''

MATH_KERNEL_TEST = '''import unittest
import math_kernel

class TestMathKernel(unittest.TestCase):
    def test_normalize(self):
        res = math_kernel.normalize([1.0, 3.0])
        self.assertEqual(res, [0.25, 0.75])

    def test_dot(self):
        self.assertEqual(math_kernel.dot_product([1, 2], [3, 4]), 11)

    def test_scale(self):
        self.assertEqual(math_kernel.compute_scale(), 86400)

if __name__ == "__main__":
    unittest.main()
'''


class DemoRunner:
    """Executes curated scenarios with deterministic outputs for live demonstrations."""

    def __init__(self, base_temp_dir: Path | str | None = None):
        self.base_dir = Path(base_temp_dir) if base_temp_dir else Path(tempfile.mkdtemp(prefix="zel_demo_"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def run_scenario_a(
        self,
        on_step: Callable[[str, dict[str, Any]], None] | None = None,
        max_generations: int = 2,
    ) -> dict[str, Any]:
        """Scenario A: Clean AST + EXEC convergence on data_pipeline.py."""
        repo_dir = self.base_dir / "scenario_a_repo"
        ingest(
            source={
                "data_pipeline.py": DATA_PIPELINE_CODE,
                "test_data_pipeline.py": DATA_PIPELINE_TEST,
            },
            repo_name=str(repo_dir),
            refresh_repo=True,
            tag="v0",
        )

        engine = SimulationEngine(
            repo_path=repo_dir,
            max_generations=max_generations,
            on_step=on_step,
        )
        final_state = engine.run()
        final_code = (repo_dir / "data_pipeline.py").read_text(encoding="utf-8") if (repo_dir / "data_pipeline.py").exists() else ""

        praises = [f for f in final_state.get("feedback_tensor", []) if f.get("verdict") == "PRAISE"]
        return {
            "scenario": "A",
            "name": "AST & EXEC Convergence",
            "status": final_state.get("status"),
            "generations": final_state.get("generation"),
            "praises": len(praises),
            "final_sha": final_state.get("main_sha"),
            "feedback_tensor": final_state.get("feedback_tensor", []),
            "final_code": final_code,
        }

    def run_scenario_b(self) -> dict[str, Any]:
        """Scenario B: Multi-hunk commit with 1 bug isolated via git bisect + ddmin."""
        repo_dir = self.base_dir / "scenario_b_repo"
        ingest(
            source={
                "math_kernel.py": MATH_KERNEL_BASE_CODE,
                "test_math_kernel.py": MATH_KERNEL_TEST,
            },
            repo_name=str(repo_dir),
            refresh_repo=True,
            tag="v0",
        )

        c0_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

        coordinator = TransformerCoordinator(repo_dir=repo_dir)
        branch_name = "evo/demo/multi-hunk-regression"
        wt = coordinator.create_worktree(branch_name=branch_name, base_sha=c0_sha)

        # Introduce 4 good hunks + 1 faulty hunk (fastmath division error)
        mutated_code = '''import math

def normalize(values):
    total = sum(values)
    # Faulty optimization: fastmath zero bypass returns NaN/empty
    return [v / 0.0 for v in values]  # fastmath division bug

def dot_product(a, b):
    # Valid optimization: list comprehension
    return sum([x * y for x, y in zip(a, b)])

def compute_scale():
    # Valid optimization: folded constant
    return 86400
'''
        (wt / "math_kernel.py").write_text(mutated_code, encoding="utf-8")
        subprocess.run(["git", "add", "math_kernel.py"], cwd=wt, check=True)
        commit_msg = (
            "exec(Kernel): multi-hunk vectorized math\n\n"
            "Agent-ID: exec-mutator-01\n"
            "Tier: EXEC\n"
            "Sub-Layer: Kernel-Optimization\n"
            "Artifact-Symbol: math_kernel\n"
            "Parent-Contract-Hash: h4sh00000000\n"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=wt, check=True)
        c1_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()

        # Run PostCommitRunner
        runner = PostCommitRunner(repo_dir=wt)
        event = runner.evaluate_commit(
            head_sha=c1_sha,
            reporter_agent="exec-mutator-01",
            reporter_tier="EXEC",
            baseline_sha=c0_sha,
        )

        grader = HardForensicGrader(repo_dir=wt)
        feedback = grader.grade(event, baseline_sha=c0_sha)

        coordinator.remove_worktree(wt, branch_name=branch_name)

        return {
            "scenario": "B",
            "name": "Multi-Hunk Bisection & ddmin",
            "baseline_sha": c0_sha,
            "bad_sha": c1_sha,
            "event_kind": event["kind"],
            "verdict": feedback["verdict"],
            "root_commit": feedback["root_commit"],
            "minimal_diff": feedback["minimal_diff"],
            "routed_tier": feedback["target_tier"],
            "causal_reason": feedback["causal_reason"],
        }

    def run_scenario_c(self) -> dict[str, Any]:
        """Scenario C: Anti-Goodhart Separation of Powers violation blocked."""
        repo_dir = self.base_dir / "scenario_c_repo"
        ingest(
            source={
                "solver.py": "class Solver:\n    def solve(self, x): return x * 2\n",
                "test_solver.py": "import unittest\nfrom solver import Solver\nclass Test(unittest.TestCase):\n    def test(self): self.assertEqual(Solver().solve(2), 4)\n",
            },
            repo_name=str(repo_dir),
            refresh_repo=True,
            tag="v0",
        )

        c0_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()
        coordinator = TransformerCoordinator(repo_dir=repo_dir)
        branch_name = "evo/demo/tamper-attempt"
        wt = coordinator.create_worktree(branch_name=branch_name, base_sha=c0_sha)

        # Mutator maliciously modifies test_solver.py to weaken contract
        (wt / "test_solver.py").write_text("import unittest\nclass Test(unittest.TestCase):\n    def test(self): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "test_solver.py"], cwd=wt, check=True)
        commit_msg = (
            "exec(Tamper): weaken assertions to pass SLA\n\n"
            "Agent-ID: exec-mutator-01\n"
            "Tier: EXEC\n"
            "Sub-Layer: Hack\n"
            "Artifact-Symbol: test_solver\n"
            "Parent-Contract-Hash: h4sh00000000\n"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=wt, check=True)
        tamper_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()

        runner = PostCommitRunner(repo_dir=wt)
        tamper_event = runner.evaluate_commit(
            head_sha=tamper_sha,
            reporter_agent="exec-mutator-01",
            reporter_tier="EXEC",
            baseline_sha=c0_sha,
        )

        grader = HardForensicGrader(repo_dir=wt)
        feedback = grader.grade(tamper_event, baseline_sha=c0_sha)

        coordinator.remove_worktree(wt, branch_name=branch_name)

        return {
            "scenario": "C",
            "name": "Anti-Goodhart Contract Guard",
            "tamper_sha": tamper_sha,
            "event_kind": tamper_event["kind"],
            "symptom_log": tamper_event["symptom_log"],
            "verdict": feedback["verdict"],
            "escalated_tier": feedback["target_tier"],
            "causal_reason": feedback["causal_reason"],
            "defense_successful": tamper_event["kind"] == "ISSUE" and feedback["target_tier"] == "SPEC",
        }

    def run_all(self) -> dict[str, Any]:
        """Run all three scenarios sequentially and return summary."""
        res_a = self.run_scenario_a(max_generations=1)
        res_b = self.run_scenario_b()
        res_c = self.run_scenario_c()
        return {
            "scenario_a": res_a,
            "scenario_b": res_b,
            "scenario_c": res_c,
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    runner = DemoRunner()
    print("[*] Running Scenario A...")
    a = runner.run_scenario_a(max_generations=1)
    print(f"  [+] Scenario A Finished: status={a['status']}, praises={a['praises']}")

    print("[*] Running Scenario B...")
    b = runner.run_scenario_b()
    print(f"  [+] Scenario B Finished: root={b['root_commit'][:8]}, diff={b['minimal_diff']}")

    print("[*] Running Scenario C...")
    c = runner.run_scenario_c()
    print(f"  [+] Scenario C Finished: defended={c['defense_successful']}, tier={c['escalated_tier']}")

    runner.cleanup()
    print("[*] All demo scenarios passed cleanly!")
