"""Zel Beta - Constraints Module.

Converts human prompts/specifications into executable Python test/constraint files
stored inside Simulation/Constraints and enforced deterministically by the Grader.
Enforces Anti-Goodhart Separation of Powers via Parent-Contract-Hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from Simulation.LLM.client import LLMClient


class ConstraintManager:
    """Manages the generation, persistence, and contract hashing of human/spec constraints."""

    def __init__(
        self,
        constraints_dir: Path | str | None = None,
        llm_client: LLMClient | None = None,
    ):
        if constraints_dir is not None:
            self.constraints_dir = Path(constraints_dir).resolve()
        else:
            self.constraints_dir = (Path(__file__).resolve().parent).resolve()

        self.constraints_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm_client or LLMClient()

    def generate_and_save(
        self,
        human_prompt: str,
        filename: str = "test_constraints.py",
        codebase_context: str | None = None,
        repo_dir: str | None = None,
    ) -> Path:
        """Translate a user prompt into a test suite and persist it into Constraints/."""
        test_code = self.llm.generate_constraint_test(
            human_prompt=human_prompt,
            codebase_context=codebase_context,
            test_module_name=filename,
            repo_dir=repo_dir,
        )

        out_path = self.constraints_dir / filename
        out_path.write_text(test_code, encoding="utf-8")
        return out_path

    def compute_contract_hash(self) -> str:
        """Compute deterministic SHA256 hash across all active constraint files.

        Used to populate Parent-Contract-Hash Git trailers and prevent code-mutating
        agents (ARCH, AST, EXEC) from weakening constraints.
        """
        hasher = hashlib.sha256()
        constraint_files = sorted(self.constraints_dir.glob("test_*.py"))
        for cf in constraint_files:
            hasher.update(cf.name.encode("utf-8"))
            hasher.update(cf.read_bytes())
        return hasher.hexdigest()[:12]

    def list_constraints(self) -> list[Path]:
        """List all active constraint test files."""
        return sorted(self.constraints_dir.glob("test_*.py"))
