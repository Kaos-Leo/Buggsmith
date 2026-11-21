"""End-to-End Integration Test for Zel Stratified Agent Swarm Simulation.

Strictly verifies Invariants:
1. Git worktree fan-out and isolation off main_sha.
2. Structured Git trailers (Agent-ID, Tier, Sub-Layer, Artifact-Symbol, Parent-Contract-Hash).
3. Post-commit Verification Runner automatically raising symptoms (Zero-LLM).
4. Hard Forensic Grader running git bisect root-cause localization & ddmin hunk minimization.
5. DPO preference dataset JSONL export.
6. Separation of powers enforcement (Anti-Goodhart).
7. Digestion packaging.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from digestion import Digestion
from ingestion import ingest
from Simulation.Constraints.generator import ConstraintManager
from Simulation.Grader import HardForensicGrader
from Simulation.Post_Commit_Runners.runner import ExperienceEvent, PostCommitRunner
from Simulation.simulation import SimulationEngine
from Simulation.Transformer.coordinator import TransformerCoordinator


class TestFullSimulationPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="zel_sim_e2e_"))
        self.repo_dir = self.temp_root / "Repo"
        self.constraints_dir = self.temp_root / "Constraints"
        self.dpo_path = self.temp_root / "training_pairs.jsonl"
        self.dist_dir = self.temp_root / "dist"

    def tearDown(self) -> None:
        # Prune worktrees before removing directory on Windows
        try:
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo_dir, capture_output=True)
        except Exception:
            pass
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_end_to_end_evolutionary_cycle(self) -> None:
        """Execute complete pipeline: Ingestion -> Constraints -> Worktree Mutation -> Verification -> Bisection & DDMin -> DPO -> Digestion."""
        # Step 1: Ingest baseline code as version 0 (C0)
        baseline_code = {
            "solver.py": (
                "class Solver:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n\n"
                "    def solve(self, items):\n"
                "        res = []\n"
                "        for x in items:\n"
                "            res.append(x * 2)\n"
                "        return res\n"
            ),
            "test_solver.py": (
                "import unittest\n"
                "from solver import Solver\n\n"
                "class TestSolver(unittest.TestCase):\n"
                "    def test_solve(self):\n"
                "        s = Solver('test')\n"
                "        self.assertEqual(s.solve([1, 2, 3]), [2, 4, 6])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        }
        ingest_res = ingest(
            source=baseline_code,
            repo_name=str(self.repo_dir),
            refresh_repo=True,
            tag="v0",
        )
        c0_sha = str(ingest_res["commit_sha"])
        self.assertTrue((self.repo_dir / ".git").exists())

        # Step 2: Constraints Manager converts user prompt into executable test
        constraint_mgr = ConstraintManager(constraints_dir=self.constraints_dir)
        constraint_file = constraint_mgr.generate_and_save(
            human_prompt="Solver must return double for input values and not mutate input list",
            filename="test_constraints.py",
        )
        self.assertTrue(constraint_file.exists())
        contract_hash = constraint_mgr.compute_contract_hash()
        self.assertTrue(len(contract_hash) >= 8)

        # Step 3: Worktree isolation & Transformer mutation
        coordinator = TransformerCoordinator(repo_dir=self.repo_dir)
        branch_name = "evo/gen1/ast-01"
        worktree_path = coordinator.create_worktree(branch_name=branch_name, base_sha=c0_sha)
        self.assertTrue(worktree_path.exists())

        # Mutate inside worktree
        rec = coordinator.mutate_file(
            file_path=worktree_path / "solver.py",
            tier="AST",
            agent_id="ast-mutator-01",
            context={"contract_hash": contract_hash, "artifact_symbol": "Solver"},
            repo_dir=worktree_path,
            auto_commit=True,
        )
        self.assertTrue(rec.success)
        c1_sha = rec.git_trailers.get("Commit-SHA")
        self.assertIsNotNone(c1_sha)

        # Verify structured git trailers in commit message
        log_res = subprocess.run(
            ["git", "log", "-1", "--format=%B", c1_sha],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Agent-ID: ast-mutator-01", log_res.stdout)
        self.assertIn("Tier: AST", log_res.stdout)
        self.assertIn(f"Parent-Contract-Hash: {contract_hash}", log_res.stdout)

        # Step 4: Post-commit Runner raises symptom automatically (PRAISE for valid optimization)
        runner = PostCommitRunner(repo_dir=worktree_path)
        praise_event = runner.evaluate_commit(
            head_sha=c1_sha,
            reporter_agent="ast-mutator-01",
            reporter_tier="AST",
            baseline_sha=c0_sha,
            test_paths=[constraint_file],
        )
        self.assertEqual(praise_event["kind"], "PRAISE")

        # Step 5: Grader processes PRAISE and logs DPO training pair
        grader = HardForensicGrader(repo_dir=worktree_path, dpo_output_path=self.dpo_path)
        feedback = grader.grade(praise_event, baseline_sha=c0_sha)
        self.assertEqual(feedback["verdict"], "PRAISE")
        self.assertEqual(feedback["root_commit"], c1_sha)
        self.assertTrue(self.dpo_path.exists())

        # Step 6: Test Forensic Bisection and DDMin on a Synthetic Regression Commit
        # Introduce a deliberate regression commit (C2) on a test branch
        broken_code = (
            "class Solver:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def solve(self, items):\n"
            "        fastmath = True\n"
            "        raise RuntimeError('fastmath NaN simulation failure')\n"
        )
        (worktree_path / "solver.py").write_text(broken_code, encoding="utf-8")
        subprocess.run(["git", "add", "solver.py"], cwd=worktree_path, check=True)
        commit_msg = (
            "exec(Vectorization): introduce fastmath kernel\n\n"
            "Agent-ID: exec-mutator-01\n"
            "Tier: EXEC\n"
            "Sub-Layer: Vectorization\n"
            f"Parent-Contract-Hash: {contract_hash}\n"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=worktree_path, check=True)
        c2_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree_path, check=True, capture_output=True, text=True).stdout.strip()

        # Runner evaluates C2 and automatically raises ISSUE symptom
        issue_event = runner.evaluate_commit(
            head_sha=c2_sha,
            reporter_agent="exec-mutator-01",
            reporter_tier="EXEC",
            baseline_sha=c1_sha,
            test_paths=[worktree_path / "test_solver.py"],
        )
        self.assertEqual(issue_event["kind"], "ISSUE")

        # Grader performs 2D Forensic Analysis: git bisect + ddmin
        issue_feedback = grader.grade(issue_event, baseline_sha=c1_sha)
        self.assertEqual(issue_feedback["verdict"], "PENALTY")
        self.assertEqual(issue_feedback["root_commit"], c2_sha)
        self.assertIn("fastmath", issue_feedback["minimal_diff"])

        # Clean up worktree
        coordinator.remove_worktree(worktree_path=worktree_path, branch_name=branch_name)

        # Step 7: Verify Separation of Powers (Anti-Goodhart Rule)
        # Attempt to have an EXEC mutator tamper with test_solver.py
        tamper_branch = "evo/tamper/exec-01"
        tamper_wt = coordinator.create_worktree(branch_name=tamper_branch, base_sha=c1_sha)
        (tamper_wt / "test_solver.py").write_text("# tampered test to bypass assertion\n", encoding="utf-8")
        subprocess.run(["git", "add", "test_solver.py"], cwd=tamper_wt, check=True)
        subprocess.run(["git", "commit", "-m", "exec(Tamper): delete tests"], cwd=tamper_wt, check=True)
        tamper_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tamper_wt, check=True, capture_output=True, text=True).stdout.strip()

        tamper_runner = PostCommitRunner(repo_dir=tamper_wt)
        tamper_event = tamper_runner.evaluate_commit(
            head_sha=tamper_sha,
            reporter_agent="exec-mutator-01",
            reporter_tier="EXEC",
            baseline_sha=c1_sha,
        )
        self.assertEqual(tamper_event["kind"], "ISSUE")
        self.assertIn("Separation of Powers Violation", tamper_event["symptom_log"])
        coordinator.remove_worktree(worktree_path=tamper_wt, branch_name=tamper_branch)

        # Step 8: Verify DPO JSONL export format (Paired preference format)
        with open(self.dpo_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        self.assertGreaterEqual(len(lines), 1)
        has_chosen = any(bool(entry.get("chosen")) for entry in lines)
        self.assertTrue(has_chosen)

        # Step 9: Digestion packages verified release
        digestion = Digestion(repo_path=self.repo_dir)
        digest_res = digestion.digest(final_version="HEAD", constraints_passed=True, output_dir=self.dist_dir)
        self.assertEqual(digest_res["status"], "accepted")
        self.assertTrue((self.dist_dir / "RELEASE_MANIFEST.json").exists())
        self.assertTrue((self.dist_dir / "solver.py").exists())

    def test_langgraph_simulation_engine_loop(self) -> None:
        """Verify the LangGraph StateGraph engine loop executes generations end-to-end."""
        baseline_code = {
            "solver.py": (
                "class Solver:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n\n"
                "    def solve(self, items):\n"
                "        res = []\n"
                "        for x in items:\n"
                "            res.append(x * 2)\n"
                "        return res\n"
            ),
            "test_solver.py": (
                "import unittest\n"
                "from solver import Solver\n\n"
                "class TestSolver(unittest.TestCase):\n"
                "    def test_solve(self):\n"
                "        s = Solver('test')\n"
                "        self.assertEqual(s.solve([1, 2, 3]), [2, 4, 6])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        }
        ingest(
            source=baseline_code,
            repo_name=str(self.repo_dir),
            refresh_repo=True,
            tag="v0",
        )

        observed_steps: list[str] = []

        def step_callback(node_name: str, state_update: dict) -> None:
            observed_steps.append(node_name)

        engine = SimulationEngine(
            repo_path=self.repo_dir,
            max_generations=2,
            constraints_dir=self.constraints_dir,
            dpo_output_path=self.dpo_path,
            on_step=step_callback,
        )

        result = engine.run()
        self.assertEqual(result["status"], "CONVERGED")
        self.assertEqual(result["generation"], 2)
        self.assertGreater(len(result["feedback_tensor"]), 0)
        self.assertIsNotNone(result["main_sha"])
        self.assertIn("transformer", observed_steps)
        self.assertIn("post_commit_runner", observed_steps)
        self.assertIn("grader", observed_steps)

    def test_parallel_fan_out_collision(self) -> None:
        """Verify parallel worktree fan-out arbitration and collision routing through Grader."""
        baseline_code = {
            "solver.py": (
                "class Solver:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n\n"
                "    def solve(self, items):\n"
                "        res = []\n"
                "        for x in items:\n"
                "            res.append(x * 2)\n"
                "        return res\n"
            ),
            "test_solver.py": (
                "import unittest\n"
                "from solver import Solver\n\n"
                "class TestSolver(unittest.TestCase):\n"
                "    def test_solve(self):\n"
                "        s = Solver('test')\n"
                "        self.assertEqual(s.solve([1, 2, 3]), [2, 4, 6])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        }
        ingest(
            source=baseline_code,
            repo_name=str(self.repo_dir),
            refresh_repo=True,
            tag="v0",
        )

        engine = SimulationEngine(
            repo_path=self.repo_dir,
            max_generations=1,
            constraints_dir=self.constraints_dir,
            dpo_output_path=self.dpo_path,
        )

        fanout_res = engine.run_parallel_fan_out_step(target_file_rel="solver.py")
        self.assertIn("worker_a_sha", fanout_res)
        self.assertIn("worker_b_sha", fanout_res)
        self.assertIn("collision_detected", fanout_res)
        self.assertIn("reconcile_packet", fanout_res)

        if fanout_res["collision_detected"] and fanout_res["reconcile_packet"]:
            self.assertEqual(fanout_res["reconcile_packet"]["verdict"], "RECONCILE")


if __name__ == "__main__":
    unittest.main()
