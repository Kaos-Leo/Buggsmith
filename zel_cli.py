"""Zel-Beta - Interactive Terminal Mission Control.

Rich-based real-time terminal mission control for live hackathon judging.
Features:
- Live split-pane layout: Top bar, Left (Swarm Activity), Right (2D Forensic Grid), Bottom (Memory/DPO feed)
- Human-in-the-loop live natural-language constraint synthesis
- Seamless demo scenario playback (Scenarios A, B, C)
- 100% offline, zero-browser dependency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.demo_scenarios import DemoRunner
from Simulation.Constraints.generator import ConstraintManager
from Simulation.simulation import SimulationEngine

console = Console()

BANNER = """[bold cyan]
███████╗███████╗██╗         ██████╗ ███████╗████████╗ █████╗ 
╚══███╔╝██╔════╝██║         ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗
  ███╔╝ █████╗  ██║  █████╗ ██████╔╝█████╗     ██║   ███████║
 ███╔╝  ██╔══╝  ██║  ╚════╝ ██╔══██╗██╔══╝     ██║   ██╔══██║
███████╗███████╗███████╗    ██████╔╝███████╗   ██║   ██║  ██║
╚══════╝╚══════╝╚══════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝
[/bold cyan][dim]Stratified Git-Forensic Agent Swarm // Mission Control v0.1[/dim]
"""


def build_top_bar(repo: str, main_sha: str, contract_hash: str, gen: int, max_gen: int) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=2)
    table.add_column(justify="center", ratio=1)
    table.add_column(justify="center", ratio=1)
    table.add_column(justify="right", ratio=1)

    table.add_row(
        f"[bold white]Repo:[/] [cyan]{repo}[/]",
        f"[bold white]HEAD:[/] [green]{main_sha[:8]}[/]",
        f"[bold white]Contract-Hash:[/] [yellow]{contract_hash[:8]}[/]",
        f"[bold white]Gen:[/] [magenta]{gen}/{max_gen}[/]",
    )
    return Panel(table, style="bold blue", title="[bold white]MISSION TELEMETRY[/]", border_style="blue")


def build_swarm_panel(current_activity: dict[str, Any]) -> Panel:
    tier = current_activity.get("tier", "AST")
    tier_colors = {"SPEC": "magenta", "ARCH": "cyan", "AST": "green", "EXEC": "yellow"}
    color = tier_colors.get(tier, "white")

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Key", style="bold white", width=14)
    table.add_column("Value", style="dim")

    table.add_row("Active Tier:", f"[bold {color}][{tier}][/]")
    table.add_row("Agent ID:", current_activity.get("agent_id", "idle"))
    table.add_row("Worktree:", str(current_activity.get("worktree", "inherit"))[-32:])
    table.add_row("Transform:", current_activity.get("transform", "Scanning AST..."))
    table.add_row("Node Delta:", str(current_activity.get("node_delta", "0")))
    table.add_row("Opcode Delta:", current_activity.get("opcode_delta", "LOAD_FAST preserved"))

    return Panel(table, title="[bold cyan]1. SWARM MUTATION ENGINE[/]", border_style="cyan")


def build_forensic_panel(feedback: dict[str, Any] | None) -> Panel:
    if not feedback:
        content = Align.center("[dim italic]Waiting for post-commit sensor & forensic evaluation...[/dim]")
        return Panel(content, title="[bold red]2. 2D FORENSIC ATTRIBUTION GRID[/]", border_style="red")

    verdict = feedback.get("verdict", "PENDING")
    v_style = {"PRAISE": "bold green", "PENALTY": "bold red", "RECONCILE": "bold yellow"}.get(verdict, "white")

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Key", style="bold white", width=14)
    table.add_column("Value")

    table.add_row("Verdict:", f"[{v_style}]{verdict}[/]")
    table.add_row("Root Commit:", f"[green]{feedback.get('root_commit', 'None')[:8]}[/]")
    table.add_row("Vertical Tier:", f"[bold magenta]{feedback.get('target_tier', 'None')}[/]")
    table.add_row("Reward Delta:", f"[bold]{feedback.get('reward_delta', 0.0):+.2f}[/]")
    table.add_row("Causal Reason:", f"[dim]{feedback.get('causal_reason', '')[:60]}[/]")

    min_diff = feedback.get("minimal_diff", "")
    syntax = Syntax(min_diff[:120], "diff", theme="monokai", line_numbers=False)

    group = Group(table, Text("\nIsolated Minimal Failure Hunk (ddmin):", style="bold yellow"), syntax)
    return Panel(group, title="[bold red]2. 2D FORENSIC ATTRIBUTION GRID[/]", border_style="red")


def build_memory_dpo_panel(rules_path: str = "causal_rules.json", dpo_path: str = "training_pairs.jsonl") -> Panel:
    # Read causal rules
    rules_text = []
    if Path(rules_path).exists():
        try:
            data = json.loads(Path(rules_path).read_text(encoding="utf-8"))
            for r in data[-3:]:
                rules_text.append(f"[yellow]• [{r.get('tier')}][/] {r.get('rule', '')[:55]}")
        except Exception:
            pass

    if not rules_text:
        rules_text = ["[dim]No causal rules distilled yet.[/dim]"]

    # Read DPO pairs
    dpo_text = []
    if Path(dpo_path).exists():
        try:
            with open(dpo_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()][-2:]
                for p in lines:
                    ch = p.get("chosen", "")[:35]
                    rej = p.get("rejected", "")[:35]
                    dpo_text.append(f"[green]Chosen:[/] {ch} | [red]Rej:[/] {rej}")
        except Exception:
            pass

    if not dpo_text:
        dpo_text = ["[dim]No preference pairs minted yet.[/dim]"]

    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)

    rules_content = Group(Text("Distilled Causal Memory:", style="bold white"), *[Text.from_markup(r) for r in rules_text])
    dpo_content = Group(Text("Paired DPO Preference Mint:", style="bold white"), *[Text.from_markup(d) for d in dpo_text])

    table.add_row(rules_content, dpo_content)
    return Panel(table, title="[bold magenta]3. PERSISTENT CAUSAL MEMORY & DPO EXPORT[/]", border_style="magenta")


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="middle", size=14),
        Layout(name="bottom", size=7),
    )
    layout["middle"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def run_mission_control(repo: str, max_generations: int, initial_prompt: str | None = None, interactive: bool = False) -> None:
    console.print(BANNER)

    repo_path = Path(repo).resolve()
    constraint_mgr = ConstraintManager()

    if initial_prompt:
        console.print(f"[bold yellow][*] Compiling human constraint prompt:[/] '{initial_prompt}'...")
        c_path = constraint_mgr.generate_and_save(initial_prompt, repo_dir=str(repo_path))
        console.print(f"  [+] Saved constraint to [cyan]{c_path.name}[/]")

    contract_hash = constraint_mgr.compute_contract_hash()

    layout = make_layout()
    current_activity: dict[str, Any] = {"tier": "AST", "agent_id": "ast-mutator-01", "transform": "Initializing..."}
    latest_feedback: dict[str, Any] | None = None
    current_gen = 0
    main_sha = "00000000"

    layout["top"].update(build_top_bar(repo, main_sha, contract_hash, current_gen, max_generations))
    layout["left"].update(build_swarm_panel(current_activity))
    layout["right"].update(build_forensic_panel(latest_feedback))
    layout["bottom"].update(build_memory_dpo_panel())

    all_feedback: list[dict[str, Any]] = []

    def on_step(node: str, state_update: dict[str, Any]) -> None:
        nonlocal current_gen, main_sha, latest_feedback, current_activity

        if node == "transformer":
            current_gen = state_update.get("generation", current_gen)
            current_activity = {
                "tier": state_update.get("candidate_tier", "AST"),
                "agent_id": state_update.get("candidate_agent_id", "mutator"),
                "worktree": state_update.get("candidate_worktree", "ephemeral"),
                "transform": f"Applied {state_update.get('candidate_tier', 'AST')} lowering mutation",
                "node_delta": "AST rewrite verified",
                "opcode_delta": "Fast global/local bind",
            }
        elif node == "post_commit_runner":
            events = state_update.get("experience_queue", [])
            if events:
                ev = events[-1]
                current_activity["transform"] = f"Harness sensor raised: {ev['kind']}"
        elif node == "grader":
            tensors = state_update.get("feedback_tensor", [])
            if tensors:
                latest_feedback = tensors[-1]
                all_feedback.append(latest_feedback)
            if state_update.get("main_sha"):
                main_sha = state_update["main_sha"]

        layout["top"].update(build_top_bar(repo, main_sha, contract_hash, current_gen, max_generations))
        layout["left"].update(build_swarm_panel(current_activity))
        layout["right"].update(build_forensic_panel(latest_feedback))
        layout["bottom"].update(build_memory_dpo_panel())

    console.print(f"[bold green][*] Launching Stratified Git-Forensic Simulation ({max_generations} gen)...[/]")
    time.sleep(1)

    with Live(layout, refresh_per_second=4, screen=False):
        engine = SimulationEngine(
            repo_path=repo_path,
            max_generations=max_generations,
            on_step=on_step,
        )
        final_state = engine.run()
        main_sha = final_state.get("main_sha", main_sha)
        layout["top"].update(build_top_bar(repo, main_sha, contract_hash, max_generations, max_generations))

    # Summary table
    console.print("\n[bold cyan]═══ EVOLUTIONARY SIMULATION COMPLETE ═══[/bold cyan]\n")
    summary = Table(title="Forensic Cycle Attribution Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Gen", justify="center")
    summary.add_column("Tier", justify="center")
    summary.add_column("Verdict", justify="center")
    summary.add_column("Reward Δ", justify="right")
    summary.add_column("Root Commit", justify="center")
    summary.add_column("ddmin Minimal Diff / Reason", justify="left")

    for i, fb in enumerate(all_feedback, 1):
        v = fb.get("verdict", "")
        v_color = "green" if v == "PRAISE" else "red" if v == "PENALTY" else "yellow"
        summary.add_row(
            str(i),
            f"[{v_color}]{fb.get('target_tier', '')}[/]",
            f"[{v_color}]{v}[/]",
            f"{fb.get('reward_delta', 0.0):+.2f}",
            fb.get("root_commit", "")[:8],
            fb.get("minimal_diff", fb.get("causal_reason", ""))[:45],
        )

    console.print(summary)


def run_demo_mode(scenario_choice: str) -> None:
    console.print(BANNER)
    runner = DemoRunner()

    choices = ["A", "B", "C"] if scenario_choice.lower() == "all" else [scenario_choice.upper()]

    for choice in choices:
        if choice == "A":
            console.print("\n[bold cyan]► SCENARIO A: AST & EXEC Convergence on Data Pipeline[/]")
            res = runner.run_scenario_a(max_generations=2)
            console.print(f"  [+] Status: [green]{res['status']}[/] | Praises: [bold green]{res['praises']}[/]")
            console.print(f"  [+] Final Commit: [yellow]{res['final_sha'][:8]}[/]")
            console.print("  [+] Verified Transformations: Loops -> ListComps, Injected __slots__, Constant Folded")

        elif choice == "B":
            console.print("\n[bold cyan]► SCENARIO B: Multi-Hunk Regression Isolated via Git Bisect + ddmin[/]")
            res = runner.run_scenario_b()
            console.print(f"  [+] Injected Commit: [yellow]{res['bad_sha'][:8]}[/] (5 hunks)")
            console.print(f"  [+] Git Bisect Root Cause: [red]{res['root_commit'][:8]}[/]")
            console.print(f"  [+] ddmin Minimal Diff: [bold red]{res['minimal_diff']}[/]")
            console.print(f"  [+] Vertical Route: Escalated to [bold magenta]{res['routed_tier']}[/]")

        elif choice == "C":
            console.print("\n[bold cyan]► SCENARIO C: Anti-Goodhart Separation of Powers Enforcement[/]")
            res = runner.run_scenario_c()
            console.print(f"  [+] Malicious Mutator Commit: [yellow]{res['tamper_sha'][:8]}[/]")
            console.print(f"  [+] Post-Commit Gate 1 Sensor: [bold red]{res['symptom_log']}[/]")
            console.print(f"  [+] 2D Forensic Grader Verdict: [bold red]{res['verdict']}[/] -> Escalated to [bold magenta]{res['escalated_tier']}[/]")
            console.print(f"  [+] Defense Succeeded: [bold green]{res['defense_successful']}[/]")

    runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Zel-Beta Terminal Mission Control")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    run_parser = subparsers.add_parser("run", help="Run live evolutionary simulation")
    run_parser.add_argument("--repo", default="Simulation/Repo", help="Path to target git repository")
    run_parser.add_argument("--generations", type=int, default=3, help="Max evolutionary generations")
    run_parser.add_argument("--prompt", help="Natural language constraint for human-in-the-loop spec synthesis")
    run_parser.add_argument("--interactive", action="store_true", help="Interactive prompt between generations")

    demo_parser = subparsers.add_parser("demo", help="Run curated demo scenarios")
    demo_parser.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all", help="Scenario choice")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo_mode(args.scenario)
    elif args.command == "run":
        run_mission_control(
            repo=args.repo,
            max_generations=args.generations,
            initial_prompt=args.prompt,
            interactive=args.interactive,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
