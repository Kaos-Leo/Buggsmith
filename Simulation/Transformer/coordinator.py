"""Zel Beta - Transformer Coordinator.

Orchestrates multi-tier mutator agents (Script, AST, Bytecode),
manages target file transformations, computes telemetry, and commits mutations
with structured Git trailers to the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from Simulation.Transformer.ast_agent import ASTLevelMutator
from Simulation.Transformer.base import BaseMutator, MutationRecord, MutationTier
from Simulation.Transformer.bytecode_agent import BytecodeLevelMutator
from Simulation.Transformer.script_agent import ScriptLevelMutator


class TransformerCoordinator:
    """Central registry and dispatch coordinator for mutator agents in Simulation/Repo."""

    def __init__(
        self,
        repo_dir: Path | str | None = None,
        agents: list[BaseMutator] | None = None,
    ):
        # Default repository path resolves to Simulation/Repo
        if repo_dir is not None:
            self.repo_dir = Path(repo_dir).resolve()
        else:
            self.repo_dir = (Path(__file__).resolve().parent.parent / "Repo").resolve()

        if agents is not None:
            self.agents = {agent.agent_id: agent for agent in agents}
        else:
            # Default three-tier swarm configuration (ARCH, AST, EXEC)
            default_agents = [
                ScriptLevelMutator(agent_id="arch-mutator-01", tier="ARCH"),
                ASTLevelMutator(agent_id="ast-mutator-01", tier="AST"),
                BytecodeLevelMutator(agent_id="exec-mutator-01", tier="EXEC"),
            ]
            self.agents = {agent.agent_id: agent for agent in default_agents}

        self.history: list[MutationRecord] = []

    def get_current_branch(self) -> str:
        """Return the currently checked out branch name in Repo."""
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    def get_head_commit(self) -> str:
        """Return HEAD commit hash in Repo."""
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    def checkout_or_create_branch(self, branch_name: str, base_ref: str | None = None) -> None:
        """Switch to branch_name in Repo, creating it from base_ref if it doesn't exist."""
        # Check if branch exists
        branches = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if branches:
            subprocess.run(["git", "checkout", branch_name], cwd=self.repo_dir, check=True, capture_output=True)
        else:
            cmd = ["git", "checkout", "-b", branch_name]
            if base_ref:
                cmd.append(base_ref)
            subprocess.run(cmd, cwd=self.repo_dir, check=True, capture_output=True)

    def create_worktree(
        self,
        branch_name: str,
        base_sha: str = "HEAD",
        worktree_path: Path | None = None,
    ) -> Path:
        """Provision an isolated git worktree off base_sha for strict parallel isolation."""
        import tempfile
        wt_dir = worktree_path or Path(tempfile.mkdtemp(prefix=f"wt_{branch_name.replace('/', '_')}_"))
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(wt_dir), base_sha],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return wt_dir

    def remove_worktree(self, worktree_path: Path, branch_name: str | None = None) -> None:
        """Cleanly remove an ephemeral worktree."""
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=self.repo_dir,
            check=False,
            capture_output=True,
        )
        if branch_name:
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=self.repo_dir,
                check=False,
                capture_output=True,
            )

    def register_agent(self, agent: BaseMutator) -> None:
        """Register a new custom mutator agent."""
        self.agents[agent.agent_id] = agent

    def _normalize_tier(self, tier: str) -> str:
        mapping = {"SCRIPT": "ARCH", "BYTECODE": "EXEC"}
        return mapping.get(str(tier).upper(), str(tier).upper())

    def get_agent_by_tier(self, tier: str) -> BaseMutator | None:
        """Find the primary registered agent for a specific tier (supports ARCH, AST, EXEC and legacy aliases)."""
        norm_tier = self._normalize_tier(tier)
        for agent in self.agents.values():
            if agent.tier == norm_tier:
                return agent
        return None

    def git_commit_mutation(self, record: MutationRecord, repo_dir: Path | None = None) -> str | None:
        """Create a Git commit in the repository recording the mutation with trailers."""
        target_repo = Path(repo_dir).resolve() if repo_dir else self.repo_dir
        try:
            rel_file = Path(record.file_path).relative_to(target_repo)
        except ValueError:
            rel_file = Path(record.file_path)

        commit_msg = record.format_commit_message()

        try:
            # Stage modified file
            subprocess.run(
                ["git", "add", str(rel_file)],
                cwd=target_repo,
                check=True,
                capture_output=True,
                text=True,
            )

            # Commit with trailers
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=target_repo,
                check=True,
                capture_output=True,
                text=True,
            )

            # Retrieve commit SHA
            sha_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=target_repo,
                check=True,
                capture_output=True,
                text=True,
            )
            sha = sha_res.stdout.strip()
            return sha
        except subprocess.CalledProcessError as e:
            record.error = f"Git commit failed: {e.stderr.strip() if e.stderr else str(e)}"
            return None

    def mutate_file(
        self,
        file_path: Path | str,
        tier: MutationTier | None = None,
        agent_id: str | None = None,
        context: dict[str, Any] | None = None,
        repo_dir: Path | None = None,
        auto_commit: bool = False,
    ) -> MutationRecord:
        """Select an agent and execute a mutation pass against target file."""
        target_repo = Path(repo_dir).resolve() if repo_dir else self.repo_dir
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = target_repo / target_path

        agent: BaseMutator | None = None
        if agent_id:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found in registry.")
        elif tier:
            agent = self.get_agent_by_tier(tier)
            if not agent:
                raise ValueError(f"No agent registered for tier '{tier}'.")
        else:
            # Default to AST mutator if unspecified
            agent = self.get_agent_by_tier("AST") or next(iter(self.agents.values()))

        record = agent.apply_to_file(file_path=target_path, context=context)
        self.history.append(record)

        if record.success and auto_commit and record.diff:
            sha = self.git_commit_mutation(record=record, repo_dir=target_repo)
            if sha:
                record.git_trailers["Commit-SHA"] = sha

        return record

    def run_multi_tier_pipeline(
        self,
        file_path: Path | str,
        tiers: list[MutationTier] | None = None,
        context: dict[str, Any] | None = None,
        repo_dir: Path | None = None,
        auto_commit_each: bool = False,
    ) -> list[MutationRecord]:
        """Run sequential mutations across multiple tiers on the same file."""
        target_repo = Path(repo_dir).resolve() if repo_dir else self.repo_dir
        target_tiers = tiers or ["SCRIPT", "AST", "BYTECODE"]
        records: list[MutationRecord] = []

        for tier in target_tiers:
            rec = self.mutate_file(
                file_path=file_path,
                tier=tier,
                context=context,
                repo_dir=target_repo,
                auto_commit=auto_commit_each,
            )
            records.append(rec)
            if not rec.success:
                # Halt pipeline if a tier introduces syntax or compilation errors
                break

        return records
