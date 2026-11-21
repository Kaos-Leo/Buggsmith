"""Zel Beta - Evolutionary Simulation Engine (LangGraph Control Plane).

Orchestrates multi-agent evolutionary code mutation, post-commit verification,
and 2D hard forensic grading.
Strictly enforces Invariants:
1. Zero raw code in LangGraph state (only lightweight SHAs, paths, and telemetry).
2. Strict worktree isolation for all worker mutations.
3. 100% deterministic, zero-LLM post-commit sensor and forensic grader.
4. Persistent causal memory feedback loop.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from Simulation.Constraints.generator import ConstraintManager
from Simulation.Grader import FeedbackPacket, HardForensicGrader
from Simulation.memory import CausalMemoryStore
from Simulation.Post_Commit_Runners.runner import ExperienceEvent, PostCommitRunner
from Simulation.Transformer.coordinator import TransformerCoordinator


class SwarmState(TypedDict):
    """LangGraph control plane state containing ONLY lightweight metadata."""

    repo_path: str
    main_sha: str
    generation: int
    max_generations: int
    candidate_sha: Optional[str]
    candidate_branch: Optional[str]
    candidate_worktree: Optional[str]
    candidate_tier: Literal["ARCH", "AST", "EXEC", "SPEC"]
    candidate_agent_id: str
    experience_queue: List[ExperienceEvent]
    feedback_tensor: List[FeedbackPacket]
    status: Literal["INITIALIZED", "RUNNING", "CONVERGED", "MAX_GEN_REACHED", "ERROR"]


class SimulationEngine:
    """Manages LangGraph workflow execution and environment lifecycle."""

    def __init__(
        self,
        repo_path: Path | str = "Simulation/Repo",
        max_generations: int = 3,
        constraints_dir: Optional[Path | str] = None,
        dpo_output_path: Path | str = "training_pairs.jsonl",
        causal_rules_path: Path | str = "causal_rules.json",
        on_step: Callable[[str, Dict[str, Any]], None] | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.max_generations = max_generations
        self.on_step = on_step
        self.coordinator = TransformerCoordinator(repo_dir=self.repo_path)
        self.constraint_mgr = ConstraintManager(constraints_dir=constraints_dir)
        self.memory_store = CausalMemoryStore(storage_path=causal_rules_path)
        self.runner = PostCommitRunner(repo_dir=self.repo_path)
        self.grader = HardForensicGrader(
            repo_dir=self.repo_path,
            dpo_output_path=dpo_output_path,
            memory_store=self.memory_store,
        )

        self.workflow = self._build_graph()

    def _get_head_sha(self) -> str:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    # Node 1: Mutator Agent Execution inside Isolated Worktree
    def transformer_node(self, state: SwarmState) -> Dict[str, Any]:
        generation = state["generation"] + 1
        main_sha = state["main_sha"]

        # Standardized tier cycle: AST -> EXEC -> ARCH
        tier_cycle: List[Tuple[Literal["ARCH", "AST", "EXEC"], str]] = [
            ("AST", "ast-mutator-01"),
            ("EXEC", "exec-mutator-01"),
            ("ARCH", "arch-mutator-01"),
        ]
        tier, agent_id = tier_cycle[(generation - 1) % len(tier_cycle)]
        branch_name = f"evo/gen{generation}/{tier.lower()}-{agent_id}"

        # Invariant 2: Provision ephemeral worktree off main_sha
        worktree_path = self.coordinator.create_worktree(branch_name=branch_name, base_sha=main_sha)

        # Context containing Parent-Contract-Hash & distilled causal rules from memory
        contract_hash = self.constraint_mgr.compute_contract_hash()
        causal_rules = self.memory_store.get_rules_for_agent(tier)

        context = {
            "contract_hash": contract_hash,
            "artifact_symbol": "target_module",
            "causal_rules": causal_rules,
        }

        # Find target python files in worktree (excluding tests and constraints)
        py_files = [f for f in worktree_path.glob("*.py") if not f.name.startswith("test_")]
        candidate_sha: Optional[str] = None

        if py_files:
            target_file = py_files[0]
            # Mutate and commit inside worktree
            rec = self.coordinator.mutate_file(
                file_path=target_file,
                tier=tier,
                agent_id=agent_id,
                context=context,
                repo_dir=worktree_path,
                auto_commit=True,
            )
            if rec.success and "Commit-SHA" in rec.git_trailers:
                candidate_sha = rec.git_trailers["Commit-SHA"]

        # Fallback if no diff was committed: pick HEAD of worktree
        if not candidate_sha:
            try:
                candidate_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except Exception:
                candidate_sha = main_sha

        return {
            "generation": generation,
            "candidate_sha": candidate_sha,
            "candidate_branch": branch_name,
            "candidate_worktree": str(worktree_path),
            "candidate_tier": tier,
            "candidate_agent_id": agent_id,
        }

    # Node 2: Post-Commit Deterministic Verification (Raises Symptom)
    def post_commit_runner_node(self, state: SwarmState) -> Dict[str, Any]:
        candidate_sha = state["candidate_sha"] or state["main_sha"]
        worktree_path = Path(state["candidate_worktree"]) if state["candidate_worktree"] else self.repo_path

        # Run verification inside the candidate worktree
        wt_runner = PostCommitRunner(repo_dir=worktree_path)
        event = wt_runner.evaluate_commit(
            head_sha=candidate_sha,
            reporter_agent=state["candidate_agent_id"],
            reporter_tier=state["candidate_tier"],
            baseline_sha=state["main_sha"],
            test_paths=self.constraint_mgr.list_constraints(),
        )

        return {
            "experience_queue": [event],
        }

    # Node 3: Hard Forensic Grader (2D Attribution Tensor)
    def grader_node(self, state: SwarmState) -> Dict[str, Any]:
        events = state.get("experience_queue", [])
        if not events:
            return {}

        event = events[-1]
        baseline_sha = state["main_sha"]
        worktree_path = Path(state["candidate_worktree"]) if state["candidate_worktree"] else self.repo_path

        # Grade event deterministically with shared persistent memory store
        wt_grader = HardForensicGrader(
            repo_dir=worktree_path,
            dpo_output_path=self.grader.dpo_path,
            memory_store=self.memory_store,
        )
        feedback = wt_grader.grade(event, baseline_sha=baseline_sha)

        new_main_sha = baseline_sha

        if feedback["verdict"] == "PRAISE":
            # Promote candidate commit to main in repo_path
            try:
                subprocess.run(
                    ["git", "merge", "--ff-only", state["candidate_sha"]],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                )
                new_main_sha = state["candidate_sha"]  # type: ignore
            except subprocess.CalledProcessError:
                pass

        # Cleanup ephemeral worktree
        if state["candidate_worktree"]:
            self.coordinator.remove_worktree(
                worktree_path=Path(state["candidate_worktree"]),
                branch_name=state["candidate_branch"],
            )

        tensor = state.get("feedback_tensor", []) + [feedback]

        return {
            "main_sha": new_main_sha,
            "feedback_tensor": tensor,
            "candidate_worktree": None,
            "candidate_branch": None,
        }

    # Conditional Routing
    def termination_router(self, state: SwarmState) -> Literal["transformer", "end"]:
        if state["generation"] >= state["max_generations"]:
            return "end"
        return "transformer"

    def _build_graph(self) -> Any:
        workflow = StateGraph(SwarmState)

        workflow.add_node("transformer", self.transformer_node)
        workflow.add_node("post_commit_runner", self.post_commit_runner_node)
        workflow.add_node("grader", self.grader_node)

        workflow.add_edge(START, "transformer")
        workflow.add_edge("transformer", "post_commit_runner")
        workflow.add_edge("post_commit_runner", "grader")

        workflow.add_conditional_edges(
            "grader",
            self.termination_router,
            {
                "transformer": "transformer",
                "end": END,
            },
        )

        return workflow.compile()

    def run_parallel_fan_out_step(
        self,
        target_file_rel: str = "solver.py",
    ) -> Dict[str, Any]:
        """Demonstrates parallel worktree fan-out off main_sha with cross-tier merge arbitration."""
        main_sha = self._get_head_sha()
        contract_hash = self.constraint_mgr.compute_contract_hash()

        # 1. Spawn Worker A (AST) in isolated worktree A
        branch_a = "evo/fanout/ast-01"
        wt_a = self.coordinator.create_worktree(branch_name=branch_a, base_sha=main_sha)
        rec_a = self.coordinator.mutate_file(
            file_path=wt_a / target_file_rel,
            tier="AST",
            agent_id="ast-mutator-01",
            context={"contract_hash": contract_hash, "artifact_symbol": "target_module"},
            repo_dir=wt_a,
            auto_commit=True,
        )

        # 2. Spawn Worker B (EXEC) in isolated worktree B
        branch_b = "evo/fanout/exec-01"
        wt_b = self.coordinator.create_worktree(branch_name=branch_b, base_sha=main_sha)
        rec_b = self.coordinator.mutate_file(
            file_path=wt_b / target_file_rel,
            tier="EXEC",
            agent_id="exec-mutator-01",
            context={"contract_hash": contract_hash, "artifact_symbol": "target_module"},
            repo_dir=wt_b,
            auto_commit=True,
        )

        # 3. Create temporary integration branch
        int_branch = "evo/fanout/integration"
        self.coordinator.checkout_or_create_branch(int_branch, base_ref=main_sha)

        # Merge Worker A
        merge_a = subprocess.run(["git", "merge", "--no-ff", "-m", "Integrate AST", branch_a], cwd=self.repo_path, capture_output=True)
        # Merge Worker B
        merge_b = subprocess.run(["git", "merge", "--no-ff", "-m", "Integrate EXEC", branch_b], cwd=self.repo_path, capture_output=True)

        collision_detected = merge_b.returncode != 0
        reconcile_packet: Optional[FeedbackPacket] = None

        if collision_detected:
            # Abort any conflicted merge state
            subprocess.run(["git", "merge", "--abort"], cwd=self.repo_path, check=False, capture_output=True)
            issue_event: ExperienceEvent = {
                "event_id": f"exp-col-{int(time.time())}",
                "kind": "ISSUE",
                "reporter_agent": "exec-mutator-01",
                "reporter_tier": "EXEC",
                "head_sha": rec_b.git_trailers.get("Commit-SHA") or main_sha,
                "predicate_cmd": "git merge --no-ff",
                "symptom_log": f"Parallel Fan-Out Git Merge Collision between {branch_a} and {branch_b}",
                "metrics": {"collision_branch_a": branch_a, "collision_branch_b": branch_b},
            }
            reconcile_packet = self.grader.grade(issue_event, baseline_sha=main_sha, is_collision=True)
        else:
            # Check for semantic collision via PostCommitRunner
            merged_sha = self._get_head_sha()
            eval_event = self.runner.evaluate_commit(
                head_sha=merged_sha,
                reporter_agent="exec-mutator-01",
                reporter_tier="EXEC",
                baseline_sha=main_sha,
                test_paths=self.constraint_mgr.list_constraints(),
            )
            if eval_event["kind"] == "ISSUE":
                collision_detected = True
                reconcile_packet = self.grader.grade(eval_event, baseline_sha=main_sha, is_collision=True)

        # Cleanup ephemeral worktrees
        self.coordinator.remove_worktree(wt_a, branch_name=branch_a)
        self.coordinator.remove_worktree(wt_b, branch_name=branch_b)
        # Return to default branch without hardcoding "master"
        try:
            default_branch = subprocess.run(
                ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            ).stdout.strip().removeprefix("origin/") or subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            ).stdout.strip() or "master"
        except Exception:
            default_branch = "master"
        self.coordinator.checkout_or_create_branch(default_branch)

        return {
            "worker_a_sha": rec_a.git_trailers.get("Commit-SHA"),
            "worker_b_sha": rec_b.git_trailers.get("Commit-SHA"),
            "collision_detected": collision_detected,
            "integration_branch": int_branch,
            "reconcile_packet": reconcile_packet,
        }

    def run(self) -> SwarmState:
        """Execute the evolutionary simulation loop end-to-end with live streaming support."""
        initial_sha = self._get_head_sha()

        initial_state: SwarmState = {
            "repo_path": str(self.repo_path),
            "main_sha": initial_sha,
            "generation": 0,
            "max_generations": self.max_generations,
            "candidate_sha": None,
            "candidate_branch": None,
            "candidate_worktree": None,
            "candidate_tier": "AST",
            "candidate_agent_id": "ast-mutator-01",
            "experience_queue": [],
            "feedback_tensor": [],
            "status": "RUNNING",
        }

        current_state = dict(initial_state)
        for output in self.workflow.stream(initial_state):
            for node_name, state_update in output.items():
                if isinstance(state_update, dict):
                    current_state.update(state_update)
                if self.on_step:
                    self.on_step(node_name, state_update if isinstance(state_update, dict) else {})

        final_state: SwarmState = current_state  # type: ignore
        final_state["status"] = "CONVERGED" if final_state["generation"] >= self.max_generations else "RUNNING"
        return final_state


if __name__ == "__main__":
    engine = SimulationEngine(max_generations=2)
    result = engine.run()
    print(f"Simulation completed! Final HEAD: {result['main_sha']} | Generations: {result['generation']}")
