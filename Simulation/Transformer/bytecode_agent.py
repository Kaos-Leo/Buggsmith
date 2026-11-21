"""Zel Beta - Intermediate / Bytecode Level Mutator Agent.

Performs execution-level optimizations and opcode tuning based on disassembly analysis.
Uses Python's `dis` module to inspect bytecode instructions, calculate opcode deltas,
and transform execution dispatch.

Supported Execution Passes:
1. CacheDecoratorPass: Injects `@functools.lru_cache` on recursive/pure calculation functions.
2. FastGlobalBindingPass: Binds frequent global lookups (`len`, `range`, `min`, `max`) to local parameters
   converting slow `LOAD_GLOBAL` opcodes into fast `LOAD_FAST` opcodes.
3. FastPathGuardPass: Injects short-circuit empty collection / boundary guards to bypass heavy bytecode loops.
"""

from __future__ import annotations

import ast
import dis
from typing import Any, ClassVar

from Simulation.Transformer.base import BaseMutator, MutationTier


class BytecodeAnalyzer:
    """Disassembles code and computes structural opcode statistics."""

    @staticmethod
    def get_opcode_stats(code_str: str) -> dict[str, Any]:
        """Compile and disassemble code string to collect instruction-level metrics."""
        try:
            compiled = compile(code_str, "<string>", "exec")
        except Exception:
            return {"total_opcodes": 0, "load_global_count": 0, "load_fast_count": 0}

        total_opcodes = 0
        load_global_count = 0
        load_fast_count = 0

        def inspect_code_obj(co: Any) -> None:
            nonlocal total_opcodes, load_global_count, load_fast_count
            for instr in dis.get_instructions(co):
                total_opcodes += 1
                if "GLOBAL" in instr.opname:
                    load_global_count += 1
                elif "FAST" in instr.opname:
                    load_fast_count += 1
            for const in co.co_consts:
                if hasattr(const, "co_code"):
                    inspect_code_obj(const)

        inspect_code_obj(compiled)
        return {
            "total_opcodes": total_opcodes,
            "load_global_count": load_global_count,
            "load_fast_count": load_fast_count,
        }


class CacheDecoratorInjector(ast.NodeTransformer):
    """Detects recursive or pure mathematical functions and injects @functools.lru_cache."""

    def __init__(self) -> None:
        self.injected_count = 0
        self.needs_functools_import = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        # Avoid decorating __init__, dunder methods, or already cached functions
        if node.name.startswith("__") and node.name.endswith("__"):
            return node

        has_cache = any(
            (isinstance(dec, ast.Name) and "cache" in dec.id.lower())
            or (isinstance(dec, ast.Attribute) and "cache" in dec.attr.lower())
            for dec in node.decorator_list
        )
        if has_cache:
            return node

        # Check if function calls itself (recursion)
        is_recursive = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == node.name:
                is_recursive = True
                break

        if is_recursive:
            cache_decorator = ast.Attribute(
                value=ast.Name(id="functools", ctx=ast.Load()),
                attr="lru_cache",
                ctx=ast.Load(),
            )
            call_decorator = ast.Call(
                func=cache_decorator,
                args=[],
                keywords=[ast.keyword(arg="maxsize", value=ast.Constant(value=128))],
            )
            node.decorator_list.insert(0, call_decorator)
            self.injected_count += 1
            self.needs_functools_import = True

        return node


class FastGlobalOptimizer(ast.NodeTransformer):
    """Converts frequent LOAD_GLOBAL opcodes in loops to LOAD_FAST by setting default args."""

    TARGET_GLOBALS: ClassVar[set[str]] = {"len", "range", "min", "max", "isinstance", "enumerate", "zip"}

    def __init__(self) -> None:
        self.optimized_functions = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        # Search for loops inside the function that use target globals
        used_globals: set[str] = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.For, ast.While)):
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Name) and sub.id in self.TARGET_GLOBALS and isinstance(sub.ctx, ast.Load):
                        used_globals.add(sub.id)

        # Check existing parameter names to prevent collision
        existing_params = {arg.arg for arg in node.args.args}
        candidates = [g for g in used_globals if f"_{g}" not in existing_params]

        if candidates:
            # Add _<g>=<g> as default parameters
            for g in candidates:
                local_name = f"_{g}"
                node.args.args.append(ast.arg(arg=local_name))
                node.args.defaults.append(ast.Name(id=g, ctx=ast.Load()))

                # Replace uses of g in function body statements with _g (leave node.args intact)
                class GlobalReplacer(ast.NodeTransformer):
                    def __init__(self, target_global: str, target_local: str) -> None:
                        self.target_global = target_global
                        self.target_local = target_local

                    def visit_Name(self, name_node: ast.Name) -> ast.Name:
                        if name_node.id == self.target_global and isinstance(name_node.ctx, ast.Load):
                            return ast.Name(id=self.target_local, ctx=ast.Load())
                        return name_node

                replacer = GlobalReplacer(target_global=g, target_local=local_name)
                node.body = [replacer.visit(stmt) for stmt in node.body]

            self.optimized_functions += 1

        return node


class FastPathGuardInjector(ast.NodeTransformer):
    """Injects short-circuit guards for functions accepting list/dict/str arguments."""

    def __init__(self) -> None:
        self.injected_guards = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if not node.args.args or node.name.startswith("__"):
            return node

        first_arg = node.args.args[0].arg
        if first_arg in ("self", "cls"):
            if len(node.args.args) < 2:
                return node
            first_arg = node.args.args[1].arg

        # Don't inject if body already starts with an If guard
        if node.body and isinstance(node.body[0], ast.If):
            return node

        # Check if first statement is a docstring; guard goes after it
        insert_idx = 1 if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)) else 0

        # Infer safe return value based on existing returns in function
        default_return_val: ast.expr = ast.Constant(value=None)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                if isinstance(sub.value, ast.List):
                    default_return_val = ast.List(elts=[], ctx=ast.Load())
                    break
                elif isinstance(sub.value, ast.Dict):
                    default_return_val = ast.Dict(keys=[], values=[])
                    break
                elif isinstance(sub.value, ast.Constant) and isinstance(sub.value.value, (int, float)):
                    default_return_val = ast.Constant(value=0)
                    break

        # Construct: if not <first_arg>: return default_return_val
        guard = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=first_arg, ctx=ast.Load())),
            body=[ast.Return(value=default_return_val)],
            orelse=[],
        )
        ast.copy_location(guard, node.body[0] if node.body else node)
        node.body.insert(insert_idx, guard)
        self.injected_guards += 1
        return node


class BytecodeLevelMutator(BaseMutator):
    """Intermediate & Bytecode Mutator optimizing execution flow and opcode density."""

    def __init__(
        self,
        agent_id: str = "exec-mutator-01",
        tier: MutationTier = "EXEC",
        sublayer: str = "Opcode-Execution-Tuning",
        enabled_passes: list[str] | None = None,
    ):
        super().__init__(agent_id=agent_id, tier=tier, sublayer=sublayer)
        self.enabled_passes = enabled_passes or [
            "cache_decorator",
            "fast_global_binding",
        ]

    def mutate_code(
        self,
        source_code: str,
        file_name: str = "module.py",
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        # Compute baseline bytecode stats
        stats_before = BytecodeAnalyzer.get_opcode_stats(source_code)

        tree = ast.parse(source_code)
        applied_actions: list[str] = []
        pass_metrics: dict[str, Any] = {}

        # Inspect persistent causal memory rules to disable conflicting passes
        active_passes = list(self.enabled_passes)
        causal_rules = (context.get("causal_rules") or []) if context else []
        disabled_passes: list[str] = []

        for rule in causal_rules:
            rule_lower = rule.lower()
            if ("fast_path" in rule_lower or "guard" in rule_lower) and "fast_path_guard" in active_passes:
                active_passes.remove("fast_path_guard")
                disabled_passes.append("fast_path_guard (causal rule)")
            if ("cache" in rule_lower or "lru" in rule_lower) and "cache_decorator" in active_passes:
                active_passes.remove("cache_decorator")
                disabled_passes.append("cache_decorator (causal rule)")
            if ("global" in rule_lower or "load_global" in rule_lower) and "fast_global_binding" in active_passes:
                active_passes.remove("fast_global_binding")
                disabled_passes.append("fast_global_binding (causal rule)")

        if disabled_passes:
            pass_metrics["disabled_by_causal_rules"] = disabled_passes

        # 1. Cache Decorator Pass
        if "cache_decorator" in active_passes:
            cache_pass = CacheDecoratorInjector()
            tree = cache_pass.visit(tree)
            if cache_pass.injected_count > 0:
                applied_actions.append(f"Injected @functools.lru_cache onto {cache_pass.injected_count} function(s)")
                pass_metrics["cache_decorators_injected"] = cache_pass.injected_count
                if cache_pass.needs_functools_import:
                    import_node = ast.Import(names=[ast.alias(name="functools")])
                    tree.body.insert(0, import_node)

        # 2. Fast Global Binding Pass (LOAD_GLOBAL -> LOAD_FAST)
        if "fast_global_binding" in active_passes:
            global_pass = FastGlobalOptimizer()
            tree = global_pass.visit(tree)
            if global_pass.optimized_functions > 0:
                applied_actions.append(f"Localized global lookups in {global_pass.optimized_functions} function(s) to convert LOAD_GLOBAL to LOAD_FAST")
                pass_metrics["functions_localized_globals"] = global_pass.optimized_functions

        # 3. Fast Path Guard Pass
        if "fast_path_guard" in active_passes:
            guard_pass = FastPathGuardInjector()
            tree = guard_pass.visit(tree)
            if guard_pass.injected_guards > 0:
                applied_actions.append(f"Injected short-circuit fast path guards into {guard_pass.injected_guards} function(s)")
                pass_metrics["fast_path_guards_injected"] = guard_pass.injected_guards

        ast.fix_missing_locations(tree)
        mutated_code = ast.unparse(tree)

        # Compute post-mutation bytecode stats
        stats_after = BytecodeAnalyzer.get_opcode_stats(mutated_code)

        opcode_delta = stats_after["total_opcodes"] - stats_before["total_opcodes"]
        global_delta = stats_after["load_global_count"] - stats_before["load_global_count"]

        if not applied_actions:
            description = "Bytecode scan completed (no bytecode dispatch bottlenecks identified)"
        else:
            description = "; ".join(applied_actions)

        metrics = {
            "opcodes_before": stats_before["total_opcodes"],
            "opcodes_after": stats_after["total_opcodes"],
            "opcode_delta": opcode_delta,
            "load_global_before": stats_before["load_global_count"],
            "load_global_after": stats_after["load_global_count"],
            "load_global_delta": global_delta,
            **pass_metrics,
        }

        return mutated_code, description, metrics
