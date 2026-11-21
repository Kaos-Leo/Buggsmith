"""Zel Beta - Script Level Mutator Agent.

Operates at the source script / algorithmic level.
Supports:
1. High-level algorithmic refactorings (linear search to set/hash lookups, f-string modernization).
2. Type annotations & contract preconditions injection.
3. Pluggable LLM prompting with automatic fallback and strict syntax safety verification.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

from Simulation.Transformer.base import BaseMutator, MutationTier


class ScriptLevelMutator(BaseMutator):
    """Script-level mutator handling algorithmic, signature, and architectural transformations."""

    def __init__(
        self,
        agent_id: str = "arch-mutator-01",
        tier: MutationTier = "ARCH",
        sublayer: str = "Algorithmic-Refactor",
        llm_callable: Callable[[str, str, dict[str, Any] | None], str] | None = None,
        strategy: str = "auto",
    ):
        super().__init__(agent_id=agent_id, tier=tier, sublayer=sublayer)
        self.llm_callable = llm_callable
        self.strategy = strategy

    def _modernize_string_formatting(self, code: str) -> tuple[str, int]:
        """Convert standard format calls or % formatting into f-strings where safe."""
        # Simple regex-based replacement for simple .format() calls
        pattern = re.compile(r'"{([^}]+)}"\.format\(([^)]+)\)')
        count = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            var_name = match.group(2).strip()
            return f'f"{{{var_name}}}"'

        new_code, num = pattern.subn(repl, code)
        return new_code, num

    def _optimize_membership_lookups(self, code: str) -> tuple[str, int]:
        """Replace membership tests against literal lists with sets for O(1) lookups."""
        tree = ast.parse(code)
        count = 0

        class SetLookupTransformer(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare) -> ast.Compare:
                nonlocal count
                self.generic_visit(node)
                new_comparators = []
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.List):
                        # Convert [1, 2, 3] to {1, 2, 3}
                        set_node = ast.Set(elts=comp.elts)
                        ast.copy_location(set_node, comp)
                        new_comparators.append(set_node)
                        count += 1
                    else:
                        new_comparators.append(comp)
                node.comparators = new_comparators
                return node

        transformed = SetLookupTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed), count

    def _inject_contract_preconditions(self, code: str, context: dict[str, Any] | None) -> tuple[str, int]:
        """Inject contract preconditions or type checks into public functions."""
        tree = ast.parse(code)
        count = 0

        class PreconditionInjector(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                nonlocal count
                self.generic_visit(node)
                if node.name.startswith("_") or not node.args.args:
                    return node

                # Only inject if function doesn't already have an assert
                has_assert = any(isinstance(stmt, ast.Assert) for stmt in node.body)
                if not has_assert:
                    first_arg = node.args.args[0].arg
                    if first_arg not in ("self", "cls"):
                        assert_node = ast.Assert(
                            test=ast.Compare(
                                left=ast.Name(id=first_arg, ctx=ast.Load()),
                                ops=[ast.IsNot()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            msg=ast.Constant(value=f"{first_arg} must not be None"),
                        )
                        node.body.insert(0, assert_node)
                        count += 1
                return node

        transformed = PreconditionInjector().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed), count

    def mutate_code(
        self,
        source_code: str,
        file_name: str = "module.py",
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        lines_before = len(source_code.splitlines())
        applied_actions: list[str] = []
        pass_metrics: dict[str, Any] = {}
        mutated_code = source_code

        # Attempt LLM mutation if callable provided or present in context
        llm = self.llm_callable or (context.get("llm_callable") if context else None)
        if llm:
            try:
                # Inject causal rules as forbidden-pattern annotation for LLM
                llm_ctx = dict(context) if context else {}
                _rules = llm_ctx.get("causal_rules") or []
                if _rules:
                    llm_ctx["_causal_rules_prompt"] = (
                        "FORBIDDEN PATTERNS (learned from prior regressions):\n"
                        + "\n".join(f"- {r}" for r in _rules)
                    )
                candidate_code = llm(source_code, file_name, llm_ctx)
                # Strip markdown code blocks if generated
                if "```python" in candidate_code:
                    candidate_code = candidate_code.split("```python")[1].split("```")[0].strip()
                elif "```" in candidate_code:
                    candidate_code = candidate_code.split("```")[1].split("```")[0].strip()

                valid, _err = self.validate_syntax(candidate_code)
                if valid and candidate_code != source_code:
                    lines_after = len(candidate_code.splitlines())
                    return (
                        candidate_code,
                        "LLM prompt-directed script refactoring",
                        {
                            "strategy": "llm_directed",
                            "lines_before": lines_before,
                            "lines_after": lines_after,
                            "line_delta": lines_after - lines_before,
                        },
                    )
            except Exception as e:
                pass_metrics["llm_error"] = str(e)

        # Inspect persistent causal memory rules to disable risky passes
        causal_rules = (context.get("causal_rules") or []) if context else []
        skip_set_lookup = any(
            any(kw in r.lower() for kw in ("set lookup", "membership", "o(1)"))
            for r in causal_rules
        )
        skip_contracts = any(
            any(kw in r.lower() for kw in ("precondition", "assert", "contract"))
            for r in causal_rules
        )
        if causal_rules:
            pass_metrics["causal_rules_active"] = len(causal_rules)

        # Also inject causal rules into LLM prompt if llm path was tried
        if causal_rules:
            pass_metrics["causal_rules_injected"] = True

        # Fallback to deterministic rule-based script transforms
        # 1. Optimize membership lookups (List -> Set in comparisons)
        if not skip_set_lookup:
            mutated_code, set_count = self._optimize_membership_lookups(mutated_code)
            if set_count > 0:
                applied_actions.append(f"Optimized {set_count} membership check(s) to O(1) set lookups")
                pass_metrics["set_lookups_optimized"] = set_count
        else:
            pass_metrics["set_lookup_disabled_by_causal_rule"] = True

        # 2. Modernize string formatting
        mutated_code, fmt_count = self._modernize_string_formatting(mutated_code)
        if fmt_count > 0:
            applied_actions.append(f"Modernized {fmt_count} string format expression(s) to f-strings")
            pass_metrics["fstrings_modernized"] = fmt_count

        # 3. Contract precondition injection (if requested in context or sublayer)
        if context and context.get("enforce_contracts", False) and not skip_contracts:
            mutated_code, contract_count = self._inject_contract_preconditions(mutated_code, context)
            if contract_count > 0:
                applied_actions.append(f"Injected {contract_count} contract precondition assertion(s)")
                pass_metrics["contract_preconditions_injected"] = contract_count

        lines_after = len(mutated_code.splitlines())
        if not applied_actions:
            description = "Script scan completed (no algorithmic script-level improvements needed)"
        else:
            description = "; ".join(applied_actions)

        metrics = {
            "strategy": "rule_based_script",
            "lines_before": lines_before,
            "lines_after": lines_after,
            "line_delta": lines_after - lines_before,
            **pass_metrics,
        }

        return mutated_code, description, metrics
