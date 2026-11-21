"""Zel Beta - Transformer Core Interfaces & Base Classes.

Provides standard contracts, telemetry records, and base mutator class
for multi-tier code transformation agents (Script, AST, Bytecode).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MutationTier = Literal["ARCH", "AST", "EXEC", "SPEC"]


@dataclass
class MutationRecord:
    """Detailed telemetry record emitted for every mutation attempt."""

    agent_id: str
    tier: MutationTier
    sublayer: str
    file_path: str
    diff: str
    description: str
    original_hash: str
    mutated_hash: str
    success: bool
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    git_trailers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_commit_message(self, summary: str | None = None) -> str:
        """Format Git commit message with structured trailers for forensic tracking."""
        title = summary or f"{self.tier.lower()}({self.sublayer}): {self.description}"
        trailers = [
            f"Agent-ID: {self.agent_id}",
            f"Tier: {self.tier}",
            f"Sub-Layer: {self.sublayer}",
        ]
        for key, value in self.git_trailers.items():
            if key not in ("Agent-ID", "Tier", "Sub-Layer"):
                trailers.append(f"{key}: {value}")

        trailer_block = "\n".join(trailers)
        return f"{title}\n\n{trailer_block}\n"


class BaseMutator(ABC):
    """Abstract base class for all mutator agents across representational tiers."""

    def __init__(self, agent_id: str, tier: MutationTier, sublayer: str):
        self.agent_id = agent_id
        self.tier = tier
        self.sublayer = sublayer

    @abstractmethod
    def mutate_code(
        self,
        source_code: str,
        file_name: str = "module.py",
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Mutate the given source code string.

        Args:
            source_code: Original Python source code.
            file_name: Filename or module identifier for context.
            context: Optional contextual parameters (feedback packets, hints, contracts).

        Returns:
            Tuple of:
                - mutated_source_code (str)
                - transformation_description (str)
                - metrics_dict (dict of int/float/str values)
        """

    def compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of code content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def compute_diff(self, original: str, mutated: str, file_name: str = "target.py") -> str:
        """Generate unified diff between original and mutated code."""
        orig_lines = original.splitlines(keepends=True)
        mut_lines = mutated.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            mut_lines,
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
        )
        return "".join(diff)

    def validate_syntax(self, code: str) -> tuple[bool, str | None]:
        """Validate whether the generated code is syntactically sound."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"

    def apply_to_file(
        self,
        file_path: Path,
        context: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> MutationRecord:
        """Read a file, mutate its contents, and optionally write changes back."""
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            return MutationRecord(
                agent_id=self.agent_id,
                tier=self.tier,
                sublayer=self.sublayer,
                file_path=str(file_path),
                diff="",
                description="Target file not found",
                original_hash="",
                mutated_hash="",
                success=False,
                error=f"File not found: {file_path}",
            )

        original_code = file_path.read_text(encoding="utf-8")
        orig_hash = self.compute_hash(original_code)

        try:
            mutated_code, description, metrics = self.mutate_code(
                original_code,
                file_name=file_path.name,
                context=context,
            )

            # Check if any change occurred
            if mutated_code == original_code:
                return MutationRecord(
                    agent_id=self.agent_id,
                    tier=self.tier,
                    sublayer=self.sublayer,
                    file_path=str(file_path),
                    diff="",
                    description=f"{description} (No modifications made)",
                    original_hash=orig_hash,
                    mutated_hash=orig_hash,
                    success=True,
                    metrics={"no_op": True, **metrics},
                )

            # Strict syntax verification
            is_valid, syntax_err = self.validate_syntax(mutated_code)
            if not is_valid:
                return MutationRecord(
                    agent_id=self.agent_id,
                    tier=self.tier,
                    sublayer=self.sublayer,
                    file_path=str(file_path),
                    diff="",
                    description="Mutation rejected due to syntax invalidity",
                    original_hash=orig_hash,
                    mutated_hash=orig_hash,
                    success=False,
                    error=syntax_err,
                )

            diff_str = self.compute_diff(original_code, mutated_code, file_path.name)
            mut_hash = self.compute_hash(mutated_code)

            if not dry_run:
                file_path.write_text(mutated_code, encoding="utf-8")

            trailers = {
                "Agent-ID": self.agent_id,
                "Tier": self.tier,
                "Sub-Layer": self.sublayer,
                "Artifact-Symbol": str(context.get("artifact_symbol", file_path.name)) if context else file_path.name,
                "Parent-Contract-Hash": str(context.get("contract_hash", "default-contract")) if context else "default-contract",
                "Original-SHA": orig_hash,
                "Mutated-SHA": mut_hash,
            }

            return MutationRecord(
                agent_id=self.agent_id,
                tier=self.tier,
                sublayer=self.sublayer,
                file_path=str(file_path),
                diff=diff_str,
                description=description,
                original_hash=orig_hash,
                mutated_hash=mut_hash,
                success=True,
                metrics=metrics,
                git_trailers=trailers,
            )

        except Exception as e:
            return MutationRecord(
                agent_id=self.agent_id,
                tier=self.tier,
                sublayer=self.sublayer,
                file_path=str(file_path),
                diff="",
                description="Unhandled mutation exception",
                original_hash=orig_hash,
                mutated_hash=orig_hash,
                success=False,
                error=str(e),
            )
