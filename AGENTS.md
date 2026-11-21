---
trigger: always_on
---

# Architectural Invariants: Stratified Git-Forensic Agent Swarm

1. **Zero Raw Code in LangGraph State:** `SwarmState` must ONLY store lightweight metadata (commit SHAs, worktree paths, agent IDs, `ExperienceEvent` structs, and `FeedbackPacket` tensors). Never pass full source files or multi-KB diffs through LangGraph state dictionaries.
2. **Strict Worktree Isolation:** Every parallel worker branch MUST execute inside an ephemeral `git worktree` created off `main_sha`. Never mutate files directly on `main` or share mutable build directories across parallel workers.
3. **Deterministic Zero-LLM Grader:** The Hard Forensic Grader (`grader/`) must remain 100% deterministic and non-LLM (`git bisect run`, `tree-sitter` AST diffing, `ddmin` delta-debugging, `pytest`, and profiler telemetry). LLMs may only act as mutation proposers inside tier subgraphs.
4. **Structured Git Trailers Required:** Every automated commit created by an agent must include machine-parseable Git trailers: `Agent-ID`, `Tier` (`ARCH` | `AST` | `EXEC` | `SPEC`), `Sub-Layer`, `Artifact-Symbol`, and `Parent-Contract-Hash`.
5. **Separation of Powers (Anti-Goodhart):** Code-mutating agents (`ARCH`, `AST`, `EXEC`) are forbidden from modifying `tests/` or `contracts/`. Only the `SPEC` tier may mutate property tests, and contract hashes are enforced deterministically before merge.
6. **Verify Before Claiming:** After creating any module, run unit and integration tests on a synthetic target repository to prove `git worktree` fan-out, `git bisect` localization, `ddmin` hunk minimization, and DPO JSONL export work end-to-end.