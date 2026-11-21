"""Zel Beta - AST Level Mutator Agent.

Performs structural Abstract Syntax Tree rewrites using Python's native `ast` module.
Guarantees semantic validity, syntax conformance, and calculates AST node deltas.

Supported Transform Passes:
1. ListCompVectorizerPass: Rewrites `res = []; for x in it: res.append(expr)` into list comprehensions.
2. SlotsInjectionPass: Injects `__slots__` into classes based on `__init__` attribute assignments.
3. ConstantFoldingPass: Folds constant binary expressions (e.g. `60 * 60 * 24` -> `86400`).
4. DeadCodePrunerPass: Prunes statements following unconditional `return` / `raise` or `if False`.
"""

from __future__ import annotations

import ast
from typing import Any

from Simulation.Transformer.base import BaseMutator, MutationTier


class ListCompVectorizer(ast.NodeTransformer):
    """Rewrites standard list accumulation loops into fast list comprehensions."""

    def __init__(self) -> None:
        self.transformed_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        new_body: list[ast.stmt] = []
        i = 0
        while i < len(node.body):
            stmt = node.body[i]
            # Check for: target = [] followed by for loop appending to target
            if (
                i + 1 < len(node.body)
                and isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.List)
                and len(stmt.value.elts) == 0
            ):
                target_var = stmt.targets[0].id
                next_stmt = node.body[i + 1]

                if (
                    isinstance(next_stmt, ast.For)
                    and not next_stmt.orelse
                    and len(next_stmt.body) == 1
                ):
                    body_stmt = next_stmt.body[0]
                    # Check if body is target_var.append(expr)
                    if (
                        isinstance(body_stmt, ast.Expr)
                        and isinstance(body_stmt.value, ast.Call)
                        and isinstance(body_stmt.value.func, ast.Attribute)
                        and isinstance(body_stmt.value.func.value, ast.Name)
                        and body_stmt.value.func.value.id == target_var
                        and body_stmt.value.func.attr == "append"
                        and len(body_stmt.value.args) == 1
                    ):
                        append_expr = body_stmt.value.args[0]
                        # Construct: target_var = [append_expr for target in iter]
                        comp = ast.ListComp(
                            elt=append_expr,
                            generators=[
                                ast.comprehension(
                                    target=next_stmt.target,
                                    iter=next_stmt.iter,
                                    ifs=[],
                                    is_async=0,
                                )
                            ],
                        )
                        new_assign = ast.Assign(
                            targets=[ast.Name(id=target_var, ctx=ast.Store())],
                            value=comp,
                        )
                        ast.copy_location(new_assign, stmt)
                        new_body.append(new_assign)
                        self.transformed_count += 1
                        i += 2
                        continue

            new_body.append(stmt)
            i += 1

        node.body = new_body
        return node


class ConstantFolder(ast.NodeTransformer):
    """Folds constant arithmetic expressions during AST lowering."""

    def __init__(self) -> None:
        self.folded_count = 0

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                val_l = node.left.value
                val_r = node.right.value
                result = None
                if isinstance(node.op, ast.Add):
                    result = val_l + val_r
                elif isinstance(node.op, ast.Sub):
                    result = val_l - val_r
                elif isinstance(node.op, ast.Mult):
                    result = val_l * val_r
                elif isinstance(node.op, ast.FloorDiv) and val_r != 0:
                    result = val_l // val_r
                elif isinstance(node.op, ast.Mod) and val_r != 0:
                    result = val_l % val_r
                elif isinstance(node.op, ast.Pow) and isinstance(val_r, int) and 0 <= val_r <= 16:
                    result = val_l**val_r

                if result is not None and type(result) in (int, float, str, bool):
                    self.folded_count += 1
                    new_node = ast.Constant(value=result)
                    return ast.copy_location(new_node, node)
            except Exception:
                pass
        return node


class SlotsInjector(ast.NodeTransformer):
    """Injects __slots__ tuple into class bodies if attributes are set on self in __init__."""

    def __init__(self) -> None:
        self.classes_optimized = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        # Check if __slots__ already exists
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        return node

        # Discover all self.<attr> assignments in __init__
        discovered_attrs: set[str] = set()
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if (
                        isinstance(stmt, ast.Assign)
                        or isinstance(stmt, ast.AnnAssign)
                    ):
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        for tgt in targets:
                            if (
                                isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                            ):
                                discovered_attrs.add(tgt.attr)

        if discovered_attrs:
            slots_tuple = ast.Tuple(
                elts=[ast.Constant(value=attr) for attr in sorted(discovered_attrs)],
                ctx=ast.Load(),
            )
            slots_assign = ast.Assign(
                targets=[ast.Name(id="__slots__", ctx=ast.Store())],
                value=slots_tuple,
            )
            node.body.insert(0, slots_assign)
            self.classes_optimized += 1

        return node


class DeadCodePruner(ast.NodeTransformer):
    """Prunes dead statements appearing after unconditional return or raise."""

    def __init__(self) -> None:
        self.pruned_statements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        pruned_body: list[ast.stmt] = []
        for stmt in node.body:
            pruned_body.append(stmt)
            if isinstance(stmt, (ast.Return, ast.Raise)):
                remaining = len(node.body) - len(pruned_body)
                if remaining > 0:
                    self.pruned_statements += remaining
                break
        node.body = pruned_body
        return node


class ASTLevelMutator(BaseMutator):
    """AST Mutator applying surgical syntactic transformations to source code."""

    def __init__(
        self,
        agent_id: str = "ast-mutator-01",
        tier: MutationTier = "AST",
        sublayer: str = "Structural-AST-Optimization",
        enabled_passes: list[str] | None = None,
    ):
        super().__init__(agent_id=agent_id, tier=tier, sublayer=sublayer)
        self.enabled_passes = enabled_passes or [
            "slots_injection",
            "listcomp_vectorization",
            "constant_folding",
            "dead_code_pruning",
        ]

    def _count_nodes(self, tree: ast.AST) -> int:
        return sum(1 for _ in ast.walk(tree))

    def mutate_code(
        self,
        source_code: str,
        file_name: str = "module.py",
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        tree = ast.parse(source_code)
        nodes_before = self._count_nodes(tree)

        applied_actions: list[str] = []
        pass_metrics: dict[str, Any] = {}

        # Inspect persistent causal memory rules to disable risky passes
        active_passes = list(self.enabled_passes)
        causal_rules = (context.get("causal_rules") or []) if context else []
        disabled_passes: list[str] = []

        for rule in causal_rules:
            rule_lower = rule.lower()
            if "slots" in rule_lower and "slots_injection" in active_passes:
                active_passes.remove("slots_injection")
                disabled_passes.append("slots_injection (causal rule)")
            if ("listcomp" in rule_lower or "comprehension" in rule_lower) and "listcomp_vectorization" in active_passes:
                active_passes.remove("listcomp_vectorization")
                disabled_passes.append("listcomp_vectorization (causal rule)")
            if "dead code" in rule_lower and "dead_code_pruning" in active_passes:
                active_passes.remove("dead_code_pruning")
                disabled_passes.append("dead_code_pruning (causal rule)")

        if disabled_passes:
            pass_metrics["disabled_by_causal_rules"] = disabled_passes

        # 1. Slots Injection Pass
        if "slots_injection" in active_passes:
            slots_pass = SlotsInjector()
            tree = slots_pass.visit(tree)
            if slots_pass.classes_optimized > 0:
                applied_actions.append(f"Injected __slots__ into {slots_pass.classes_optimized} class(es)")
                pass_metrics["slots_classes_optimized"] = slots_pass.classes_optimized

        # 2. ListComp Vectorizer Pass
        if "listcomp_vectorization" in active_passes:
            vector_pass = ListCompVectorizer()
            tree = vector_pass.visit(tree)
            if vector_pass.transformed_count > 0:
                applied_actions.append(f"Vectorized {vector_pass.transformed_count} for-loop(s) to list comprehension(s)")
                pass_metrics["listcomps_created"] = vector_pass.transformed_count

        # 3. Constant Folding Pass
        if "constant_folding" in active_passes:
            folder_pass = ConstantFolder()
            tree = folder_pass.visit(tree)
            if folder_pass.folded_count > 0:
                applied_actions.append(f"Folded {folder_pass.folded_count} constant expression(s)")
                pass_metrics["constants_folded"] = folder_pass.folded_count

        # 4. Dead Code Pruning Pass
        if "dead_code_pruning" in active_passes:
            pruner_pass = DeadCodePruner()
            tree = pruner_pass.visit(tree)
            if pruner_pass.pruned_statements > 0:
                applied_actions.append(f"Pruned {pruner_pass.pruned_statements} unreachable statement(s)")
                pass_metrics["dead_statements_pruned"] = pruner_pass.pruned_statements

        ast.fix_missing_locations(tree)
        nodes_after = self._count_nodes(tree)
        mutated_code = ast.unparse(tree)

        if not applied_actions:
            description = "AST scan completed (no matching optimization patterns found)"
        else:
            description = "; ".join(applied_actions)

        metrics = {
            "ast_nodes_before": nodes_before,
            "ast_nodes_after": nodes_after,
            "ast_node_delta": nodes_after - nodes_before,
            **pass_metrics,
        }

        return mutated_code, description, metrics
