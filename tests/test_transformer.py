"""Unit and Integration Tests for Zel Transformer Multi-Tier Mutation Engine."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from Simulation.Transformer import (
    ASTLevelMutator,
    BytecodeLevelMutator,
    ScriptLevelMutator,
    TransformerCoordinator,
)


class TestTransformerAgents(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="zel_transformer_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ast_mutator_listcomp_and_slots(self) -> None:
        """Verify AST mutator correctly vectorizes loops and injects __slots__."""
        source_code = (
            "class NodeState:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n\n"
            "def compute_squares(numbers):\n"
            "    out = []\n"
            "    for n in numbers:\n"
            "        out.append(n * n)\n"
            "    return out\n"
        )
        mutator = ASTLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code)

        # Check __slots__ injected
        self.assertIn("__slots__ = ('x', 'y')", mutated)
        self.assertIn("out = [n * n for n in numbers]", mutated)
        self.assertGreater(metrics["slots_classes_optimized"], 0)
        self.assertGreater(metrics["listcomps_created"], 0)
        self.assertEqual(metrics["ast_nodes_before"], 49)
        self.assertEqual(metrics["ast_nodes_after"], 49)

    def test_ast_mutator_constant_folding(self) -> None:
        """Verify AST mutator folds constant arithmetic."""
        source_code = (
            "def get_day_seconds():\n"
            "    return 60 * 60 * 24\n"
        )
        mutator = ASTLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code)

        self.assertIn("return 86400", mutated)
        self.assertGreater(metrics["constants_folded"], 0)

    def test_bytecode_mutator_cache_and_global_binding(self) -> None:
        """Verify Bytecode mutator injects @lru_cache and binds globals."""
        source_code = (
            "def fibonacci(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
            "def iterate_items(items):\n"
            "    total = 0\n"
            "    for i in range(len(items)):\n"
            "        total += items[i]\n"
            "    return total\n"
        )
        mutator = BytecodeLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code)

        # Check @lru_cache injected on recursive function
        self.assertIn("functools.lru_cache", mutated)
        self.assertIn("import functools", mutated)
        # Check global binding for len/range
        self.assertIn("_range=range", mutated)
        self.assertIn("_len=len", mutated)
        self.assertIn("opcodes_before", metrics)
        self.assertIn("opcodes_after", metrics)

    def test_script_mutator_set_lookup(self) -> None:
        """Verify Script mutator optimizes list membership to O(1) set membership."""
        source_code = (
            "def check_valid(flag):\n"
            "    if flag in [1, 2, 3, 4, 5]:\n"
            "        return True\n"
            "    return False\n"
        )
        mutator = ScriptLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code)

        self.assertIn("{1, 2, 3, 4, 5}", mutated)
        self.assertEqual(metrics["set_lookups_optimized"], 1)

    def test_ast_dead_code_pruning(self) -> None:
        """Verify AST mutator removes statements following return or raise."""
        source_code = (
            "def process_data(data):\n"
            "    return len(data)\n"
            "    print('unreachable statement')\n"
            "    x = 10\n"
        )
        mutator = ASTLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code)

        self.assertNotIn("unreachable statement", mutated)
        self.assertEqual(metrics["dead_statements_pruned"], 2)

    def test_script_contract_preconditions(self) -> None:
        """Verify Script mutator can inject defensive assertions if requested."""
        source_code = (
            "def handle_request(payload):\n"
            "    return payload.get('action')\n"
        )
        mutator = ScriptLevelMutator()
        mutated, _desc, metrics = mutator.mutate_code(source_code, context={"enforce_contracts": True})

        self.assertIn("assert payload is not None", mutated)
        self.assertEqual(metrics["contract_preconditions_injected"], 1)

    def test_syntax_safety_rejection(self) -> None:
        """Verify mutators reject invalid code and never write corrupt files."""
        test_file = self.temp_dir / "broken.py"
        valid_code = "x = 42\n"
        test_file.write_text(valid_code, encoding="utf-8")

        # Mutator producing bad syntax
        mutator = ScriptLevelMutator()
        mutator.mutate_code = lambda code, file_name="target.py", context=None: ("def bad_syntax(:\n", "broken", {})  # type: ignore

        rec = mutator.apply_to_file(test_file)
        self.assertFalse(rec.success)
        self.assertIn("SyntaxError", rec.error)
        # Content remains uncorrupted
        self.assertEqual(test_file.read_text(encoding="utf-8"), valid_code)

    def test_default_repo_path(self) -> None:
        """Verify coordinator defaults to Simulation/Repo."""
        coordinator = TransformerCoordinator()
        expected_repo = (Path(__file__).resolve().parent.parent / "Simulation" / "Repo").resolve()
        self.assertEqual(coordinator.repo_dir, expected_repo)

    def test_multi_tier_pipeline_sequence(self) -> None:
        """Verify sequential pipeline execution: SCRIPT -> AST -> BYTECODE."""
        test_file = self.temp_dir / "pipeline_target.py"
        test_file.write_text(
            "class Tracker:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n\n"
            "def filter_evens(nums):\n"
            "    if 1 in [2, 3]:\n"
            "        pass\n"
            "    evens = []\n"
            "    for n in nums:\n"
            "        evens.append(n)\n"
            "    return evens\n",
            encoding="utf-8",
        )

        coordinator = TransformerCoordinator()
        records = coordinator.run_multi_tier_pipeline(test_file)

        self.assertEqual(len(records), 3)
        self.assertTrue(all(r.success for r in records))

        final_content = test_file.read_text(encoding="utf-8")
        # Script tier should have converted [2, 3] to {2, 3}
        self.assertIn("{2, 3}", final_content)
        # AST tier should have injected slots and vectorized loop
        self.assertIn("__slots__", final_content)
        self.assertIn("evens = [n for n in nums]", final_content)

    def test_coordinator_pipeline_and_git_commit(self) -> None:
        """Test multi-tier pipeline and git commit with structured trailers."""
        # Initialize an isolated git repository in temp_dir
        subprocess.run(["git", "init"], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "mutator@zel.internal"],
            cwd=self.temp_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Zel Transformer Agent"],
            cwd=self.temp_dir,
            check=True,
            capture_output=True,
        )

        test_file = self.temp_dir / "target.py"
        test_file.write_text(
            "def calculate(items):\n"
            "    res = []\n"
            "    for x in items:\n"
            "        res.append(x * 2)\n"
            "    return res\n",
            encoding="utf-8",
        )

        # Commit initial baseline
        subprocess.run(["git", "add", "target.py"], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial baseline"], cwd=self.temp_dir, check=True, capture_output=True)

        coordinator = TransformerCoordinator()
        record = coordinator.mutate_file(
            file_path=test_file,
            tier="AST",
            repo_dir=self.temp_dir,
            auto_commit=True,
        )

        self.assertTrue(record.success)
        self.assertIn("Commit-SHA", record.git_trailers)

        # Verify git log trailers
        log_res = subprocess.run(
            ["git", "log", "-1", "--pretty=fuller"],
            cwd=self.temp_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Agent-ID: ast-mutator-01", log_res.stdout)
        self.assertIn("Tier: AST", log_res.stdout)


if __name__ == "__main__":
    unittest.main()
