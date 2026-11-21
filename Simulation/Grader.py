"""Zel Beta - Hard Forensic Grader.

Operates across a 2D Attribution Grid:
- Horizontal: Traces causality backward along the Git commit DAG (git bisect + true ddmin delta-debugging).
- Vertical: Traces causality across abstraction tiers (SPEC > ARCH > AST > EXEC) via deterministic escalation rules.
- Paired DPO / GRPO export: Emits verified preference pairs (prompt, chosen, rejected) to training_pairs.jsonl.
- Causal Rule Store: Connects to CausalMemoryStore to persist learned rules across generations.
100% Zero-LLM: Purely deterministic and symbolic.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, TypedDict

from Simulation.memory import CausalMemoryStore
from Simulation.Post_Commit_Runners.runner import ExperienceEvent


class FeedbackPacket(TypedDict):
    """Structured 2D attribution tensor emitted for each graded cycle."""

    verdict: Literal["PENALTY", "PRAISE", "RECONCILE"]
    target_tier: Literal["ARCH", "AST", "EXEC", "SPEC"]
    target_sublayer: str
    target_agent: str
    root_commit: str
    enabling_commits: List[str]
    minimal_diff: str
    causal_reason: str
    reward_delta: float
    timestamp: str


class GitTrailersParser:
    """Parses machine-readable trailers from git commit messages."""

    @staticmethod
    def parse_commit_trailers(repo_dir: Path, commit_sha: str) -> Dict[str, str]:
        try:
            res = subprocess.run(
                ["git", "log", "-1", "--format=%B", commit_sha],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            trailers: Dict[str, str] = {}
            for line in res.stdout.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key in ("Agent-ID", "Tier", "Sub-Layer", "Artifact-Symbol", "Parent-Contract-Hash"):
                        trailers[key] = val
            return trailers
        except subprocess.CalledProcessError:
            return {}


class HorizontalGrader:
    """Horizontal causality tracing: git bisect localization and real ddmin hunk delta-debugging."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir

    def _get_env(self) -> Dict[str, str]:
        """Ensure subprocesses run without writing untracked bytecode (.pyc) files."""
        return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def bisect_root_cause(
        self,
        bad_sha: str,
        good_sha: str,
        predicate_cmd: str,
    ) -> str:
        """Run automated git bisect to isolate the first bad commit in O(log N) steps."""
        if bad_sha == good_sha:
            return bad_sha

        env = self._get_env()

        # Fast path: check if bad_sha is a single commit on top of good_sha
        try:
            rev_list = subprocess.run(
                ["git", "rev-list", f"{good_sha}..{bad_sha}"],
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip().splitlines()
            if len(rev_list) == 1:
                return rev_list[0]
        except subprocess.CalledProcessError:
            pass

        # Execute git bisect run
        try:
            subprocess.run(["git", "bisect", "reset"], check=False, cwd=self.repo_dir, capture_output=True, env=env)
            subprocess.run(["git", "bisect", "start", bad_sha, good_sha], check=True, cwd=self.repo_dir, capture_output=True, env=env)

            bisect_run = subprocess.run(
                f"git bisect run {predicate_cmd}",
                check=False,
                cwd=self.repo_dir,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
            )
            first_bad = bad_sha
            for line in bisect_run.stdout.splitlines():
                if "is the first bad commit" in line:
                    first_bad = line.split()[0]
                    break

            subprocess.run(["git", "bisect", "reset"], check=False, cwd=self.repo_dir, capture_output=True, env=env)
            return first_bad
        except Exception:
            subprocess.run(["git", "bisect", "reset"], check=False, cwd=self.repo_dir, capture_output=True, env=env)
            return bad_sha

    def _split_diff_into_scoped_hunks(self, diff_text: str) -> List[Tuple[str, str]]:
        """Split a multi-file diff into (file_header, hunk_body) tuples.

        Guarantees that each hunk is paired with its corresponding file's
        diff/index/---/+++ header block, so multi-file patches apply cleanly.
        """
        results: List[Tuple[str, str]] = []
        current_header_lines: List[str] = []
        current_hunk_lines: List[str] = []
        in_hunk = False

        for line in diff_text.splitlines(keepends=True):
            if line.startswith("diff --git"):
                if in_hunk and current_hunk_lines:
                    results.append(("".join(current_header_lines), "".join(current_hunk_lines)))
                    current_hunk_lines = []
                    in_hunk = False
                current_header_lines = [line]
            elif line.startswith("@@"):
                if in_hunk and current_hunk_lines:
                    results.append(("".join(current_header_lines), "".join(current_hunk_lines)))
                    current_hunk_lines = []
                in_hunk = True
                current_hunk_lines.append(line)
            else:
                if in_hunk:
                    current_hunk_lines.append(line)
                else:
                    current_header_lines.append(line)

        if in_hunk and current_hunk_lines:
            results.append(("".join(current_header_lines), "".join(current_hunk_lines)))

        return results

    def _reconstruct_patch(self, scoped_hunks: List[Tuple[str, str]]) -> str:
        """Reconstruct unified diff text from a subset of scoped hunks."""
        output: List[str] = []
        current_hdr: Optional[str] = None
        for hdr, hunk in scoped_hunks:
            if hdr != current_hdr:
                output.append(hdr)
                current_hdr = hdr
            output.append(hunk)
        return "".join(output)

    def ddmin_minimal_diff(self, commit_sha: str, predicate_cmd: str) -> str:
        """True delta-debugging (ddmin): partitions diff hunks to isolate the minimal causal failure.

        Strategy:
        1. Obtain unified diff of commit_sha vs its parent.
        2. Split into scoped hunks (file_header, hunk_body).
        3. For multi-hunk diffs: binary-splitting ddmin loop using real `git apply` + predicate execution
           to determine which hunk subset reproduces the failure (returncode != 0 == failure).
        4. Single-hunk or git-apply failure: keyword scan on added lines as fallback.
        """
        env = self._get_env()
        try:
            diff_res = subprocess.run(
                ["git", "show", "--format=", "--unified=0", commit_sha],
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            diff_text = diff_res.stdout.strip()
            if not diff_text:
                return f"Commit {commit_sha} diff"

            scoped_hunks = self._split_diff_into_scoped_hunks(diff_text)
            parent_sha = f"{commit_sha}~1"

            def _keyword_scan(text: str) -> str:
                """Fallback: return the most suspicious added line from the diff text."""
                lines = [l for l in text.splitlines() if l.startswith("+") and not l.startswith("+++")]
                for kw in ("fastmath", "raise", "error", "nan", "none", "assert"):
                    for l in lines:
                        if kw in l.lower():
                            return l
                return lines[0] if lines else text[:120]

            if len(scoped_hunks) <= 1:
                return _keyword_scan(diff_text)

            # Binary-splitting ddmin loop over scoped hunks using real predicate execution
            def test_subset(subset: List[Tuple[str, str]]) -> bool:
                """Return True if applying only this hunk subset reproduces the failure (exit != 0)."""
                patch_text = self._reconstruct_patch(subset)
                try:
                    # Reset to parent state
                    subprocess.run(
                        ["git", "checkout", parent_sha, "--", "."],
                        cwd=self.repo_dir,
                        check=False,
                        capture_output=True,
                        env=env,
                    )
                    # Try to apply the subset patch
                    apply_res = subprocess.run(
                        ["git", "apply", "--check", "-"],
                        input=patch_text,
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if apply_res.returncode != 0:
                        # Patch doesn't apply cleanly — treat as "doesn't reproduce"
                        return False
                    subprocess.run(
                        ["git", "apply", "-"],
                        input=patch_text,
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        env=env,
                    )
                    # Run predicate: failure (non-zero) == reproduces the bug
                    pred_res = subprocess.run(
                        predicate_cmd,
                        shell=True,
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    return pred_res.returncode != 0
                except Exception:
                    return False
                finally:
                    # Always restore to the original bad commit state
                    subprocess.run(
                        ["git", "checkout", commit_sha, "--", "."],
                        cwd=self.repo_dir,
                        check=False,
                        capture_output=True,
                        env=env,
                    )

            n = 2
            c = list(scoped_hunks)

            while len(c) >= 2:
                subsets: List[List[Tuple[str, str]]] = []
                step = max(1, len(c) // n)
                for i in range(0, len(c), step):
                    subsets.append(c[i : i + step])

                some_sub_failed = False
                for sub in subsets:
                    if test_subset(sub):
                        c = sub
                        n = 2
                        some_sub_failed = True
                        break

                if not some_sub_failed:
                    if n < len(c):
                        n = min(len(c), n * 2)
                    else:
                        break

            minimal_patch = self._reconstruct_patch(c).strip()
            return _keyword_scan(minimal_patch) if minimal_patch else f"Hunk from {commit_sha}"

        except Exception:
            return f"Diff for {commit_sha}"

    def propagate_credit(self, leaf_sha: str, base_reward: float = 1.0, decay: float = 0.5) -> List[Tuple[str, float]]:
        """Calculate exponentially decayed enabling credit along commit ancestors."""
        credits: List[Tuple[str, float]] = [(leaf_sha, base_reward)]
        env = self._get_env()
        try:
            parents = subprocess.run(
                ["git", "rev-parse", f"{leaf_sha}^@"],
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip().splitlines()

            reward = base_reward
            for p in parents:
                if p:
                    reward *= decay
                    credits.append((p, round(reward, 3)))
        except subprocess.CalledProcessError:
            pass
        return credits


class VerticalGrader:
    """Vertical causality tracing across abstraction tiers (SPEC > ARCH > AST > EXEC)."""

    TIER_PRECEDENCE: ClassVar[Dict[str, int]] = {"SPEC": 4, "ARCH": 3, "AST": 2, "EXEC": 1}

    @staticmethod
    def normalize_tier(tier: str) -> Literal["ARCH", "AST", "EXEC", "SPEC"]:
        mapping = {"SCRIPT": "ARCH", "BYTECODE": "EXEC"}
        norm = mapping.get(tier.upper(), tier.upper())
        if norm in ("ARCH", "AST", "EXEC", "SPEC"):
            return norm  # type: ignore
        return "AST"

    @classmethod
    def route_symptom(
        cls,
        event: ExperienceEvent,
        commit_trailers: Dict[str, str],
    ) -> Tuple[Literal["ARCH", "AST", "EXEC", "SPEC"], str, str]:
        """Determine target tier, root violation type, and causal rule according to escalation rules."""
        symptom = event["symptom_log"].lower()
        detected_tier = cls.normalize_tier(event["reporter_tier"])
        target_tier = cls.normalize_tier(commit_trailers.get("Tier", detected_tier))

        # Rule 1: Contract integrity / separation of powers
        if "contract" in symptom or "separation of powers" in symptom:
            return "SPEC", "Contract Boundary Invariant Broken", "Code-mutating tiers must not modify contracts/ or tests/."

        # Rule 2: Execution compiler/vectorization failure due to data layout -> Escalate to AST
        if (detected_tier == "EXEC" or target_tier == "EXEC") and (
            "cannot vectorize" in symptom or "typeerror" in symptom or "layout" in symptom or "numba" in symptom
        ):
            return "AST", "Data Layout Impeding Lowering", "Escalated to AST tier: Refactor data structure into contiguous layout."

        # Rule 3: AST memory/serialization failure -> Escalate to ARCH
        if (detected_tier == "AST" or target_tier == "AST") and (
            "memory budget" in symptom or "serialization" in symptom or "module" in symptom
        ):
            return "ARCH", "Module Boundary Memory Invariant", "Escalated to ARCH tier: Redesign module decomposition."

        # Rule 4: Property test catches numeric or signed-zero regression under fastmath
        if "assertionerror" in symptom and "fastmath" in symptom:
            return "EXEC", "IEEE-754 Invariant Broken", "Disable fastmath on floating-point reduction kernels."

        return target_tier, "Local Assertion/Test Regression", f"Fix regression in {target_tier} transformation."


class HardForensicGrader:
    """Complete 2D Forensic Grader: Horizontal Lineage + Vertical Tier Escalation + Paired DPO Minting."""

    def __init__(
        self,
        repo_dir: Path | str = "Simulation/Repo",
        dpo_output_path: Path | str = "training_pairs.jsonl",
        memory_store: Optional[CausalMemoryStore] = None,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.dpo_path = Path(dpo_output_path).resolve()
        self.horizontal = HorizontalGrader(self.repo_dir)
        self.vertical = VerticalGrader()
        self.memory_store = memory_store or CausalMemoryStore()
        # In-memory buffer of pending rejections for paired DPO generation: keyed by artifact_symbol
        self.pending_rejections: Dict[str, Dict[str, Any]] = {}

    def grade(
        self,
        event: ExperienceEvent,
        baseline_sha: str,
        is_collision: bool = False,
    ) -> FeedbackPacket:
        """Process an ExperienceEvent and emit a 2D FeedbackPacket."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if event["kind"] == "ISSUE":
            # 1. Horizontal: Isolate root commit via git bisect
            root_commit = self.horizontal.bisect_root_cause(
                bad_sha=event["head_sha"],
                good_sha=baseline_sha,
                predicate_cmd=event["predicate_cmd"],
            )

            # 2. Horizontal: Extract minimal diff via real ddmin
            minimal_diff = self.horizontal.ddmin_minimal_diff(
                commit_sha=root_commit,
                predicate_cmd=event["predicate_cmd"],
            )

            trailers = GitTrailersParser.parse_commit_trailers(self.repo_dir, root_commit)
            target_agent = trailers.get("Agent-ID", event["reporter_agent"])
            target_tier = self.vertical.normalize_tier(trailers.get("Tier", event["reporter_tier"]))
            sublayer = trailers.get("Sub-Layer", "General")
            symbol = trailers.get("Artifact-Symbol", "target_symbol")

            # 3. Vertical: Route across abstraction tiers
            routed_tier, violation_type, causal_rule = self.vertical.route_symptom(event, trailers)

            verdict: Literal["PENALTY", "RECONCILE"] = "RECONCILE" if is_collision else "PENALTY"
            reward_delta = 0.0 if is_collision else -0.6

            packet: FeedbackPacket = {
                "verdict": verdict,
                "target_tier": routed_tier,
                "target_sublayer": sublayer,
                "target_agent": target_agent,
                "root_commit": root_commit,
                "enabling_commits": [],
                "minimal_diff": minimal_diff,
                "causal_reason": f"{violation_type}: {causal_rule}",
                "reward_delta": reward_delta,
                "timestamp": timestamp,
            }

            # Store distilled rule in persistent memory store
            self.memory_store.add_rule(
                tier=routed_tier,
                sublayer=sublayer,
                rule=causal_rule,
                commit=root_commit,
                minimal_diff=minimal_diff,
                artifact_symbol=symbol,
            )

            # Buffer rejection for subsequent paired DPO record
            self.pending_rejections[symbol] = {
                "commit": root_commit,
                "minimal_diff": minimal_diff,
                "failure_symptom": event["symptom_log"][:160],
                "tier": routed_tier,
                "causal_rule": causal_rule,
                "predicate_cmd": event["predicate_cmd"],
            }

            return packet

        else:
            # Event is PRAISE: Reward leaf commit and propagate lineage credit
            trailers = GitTrailersParser.parse_commit_trailers(self.repo_dir, event["head_sha"])
            target_agent = trailers.get("Agent-ID", event["reporter_agent"])
            target_tier = self.vertical.normalize_tier(trailers.get("Tier", event["reporter_tier"]))
            sublayer = trailers.get("Sub-Layer", "Optimization")
            symbol = trailers.get("Artifact-Symbol", "target_symbol")

            credits = self.horizontal.propagate_credit(event["head_sha"], base_reward=1.0, decay=0.5)
            enabling = [sha for sha, _ in credits[1:]]

            packet: FeedbackPacket = {
                "verdict": "PRAISE",
                "target_tier": target_tier,
                "target_sublayer": sublayer,
                "target_agent": target_agent,
                "root_commit": event["head_sha"],
                "enabling_commits": enabling,
                "minimal_diff": "All constraints verified cleanly",
                "causal_reason": "Deterministic checks passed and performance targets satisfied.",
                "reward_delta": 1.0,
                "timestamp": timestamp,
            }

            # Emit PAIRED DPO Record if there was a prior rejection for this symbol
            if symbol in self.pending_rejections:
                prior = self.pending_rejections.pop(symbol)
                self._mint_dpo_pair(
                    prompt=f"Optimize {symbol} satisfying {prior.get('predicate_cmd', 'tests')}",
                    rejected=f"Commit: {prior['commit']} | Minimal Diff: {prior['minimal_diff']} | Error: {prior['failure_symptom']}",
                    chosen=f"Commit: {event['head_sha']} | Agent: {target_agent} ({target_tier})",
                    metadata={
                        "verdict": "PAIRED_DPO",
                        "symbol": symbol,
                        "chosen_tier": target_tier,
                        "rejected_tier": prior["tier"],
                        "causal_rule": prior.get("causal_rule", ""),
                    },
                )
            else:
                # Standalone positive milestone
                self._mint_dpo_pair(
                    prompt=f"Optimize {symbol} maintaining invariants",
                    rejected="",
                    chosen=f"Commit: {event['head_sha']} | Agent: {target_agent} ({target_tier})",
                    metadata={"verdict": "PRAISE_SOLO", "credits": credits},
                )

            return packet

    def _mint_dpo_pair(
        self,
        prompt: str,
        rejected: str,
        chosen: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Write paired self-training preference tuple to training_pairs.jsonl."""
        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.dpo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dpo_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
