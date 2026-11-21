"""Zel Beta - Command Line Interface & Terminal Visualizer.

Provides interactive execution and visual inspection of the Stratified Agent Swarm.
Uses rich terminal tables and panels to render the 2D Attribution Grid, active worktrees,
causal memory rules, and DPO training pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from Simulation.Constraints.generator import ConstraintManager
from Simulation.memory import CausalMemoryStore
from Simulation.simulation import SimulationEngine

console = Console()


def display_banner() -> None:
    banner = Text(
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        "║            ZEL-BETA: STRATIFIED GIT-FORENSIC AGENT SWARM            ║\n"
        "║   Control Plane: LangGraph | Sensor: Zero-LLM | Grader: 2D Forensics ║\n"
        "╚══════════════════════════════════════════════════════════════════════╝",
        style="bold cyan",
    )
    console.print(banner)


def cmd_run(args: argparse.Namespace) -> None:
    display_banner()
    console.print(f"[bold green][*] Initializing SimulationEngine (Max Generations: {args.generations})...[/bold green]")

    engine = SimulationEngine(
        repo_path=args.repo,
        max_generations=args.generations,
    )

    console.print(f"[dim]Base Repository: {engine.repo_path}[/dim]")
    console.print("[yellow][*] Starting Evolutionary Simulation Loop...[/yellow]")

    result = engine.run()

    console.print("\n[bold green]═══ SIMULATION CYCLE COMPLETE ═══[/bold green]")
    console.print(f"[cyan]Final HEAD Commit: [bold]{result['main_sha']}[/bold][/cyan]")
    console.print(f"[cyan]Total Generations: {result['generation']}[/cyan]")
    console.print(f"[cyan]Status: {result['status']}[/cyan]\n")

    # Display 2D Feedback Tensor Table
    table = Table(title="2D Attribution Grid (Feedback Tensor)", show_header=True, header_style="bold magenta")
    table.add_column("Gen", style="dim", width=4)
    table.add_column("Target Tier", style="bold yellow")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Root Commit", style="green")
    table.add_column("Minimal Diff (ddmin)", style="dim")
    table.add_column("Causal Reason / Rule", style="white")

    tensor = result.get("feedback_tensor", [])
    for idx, packet in enumerate(tensor, 1):
        verdict = packet["verdict"]
        verdict_style = "bold green" if verdict == "PRAISE" else ("bold yellow" if verdict == "RECONCILE" else "bold red")
        table.add_row(
            str(idx),
            packet["target_tier"],
            packet["target_agent"],
            f"[{verdict_style}]{verdict}[/{verdict_style}]",
            packet["root_commit"][:7],
            packet["minimal_diff"][:30],
            packet["causal_reason"][:45] + ("..." if len(packet["causal_reason"]) > 45 else ""),
        )

    console.print(table)

    # Display Causal Rules Learned
    rules = engine.memory_store.get_all_rules()
    if rules:
        rule_panel = Table(title="Persistent Causal Memory (Active Rules)", show_header=True, header_style="bold blue")
        rule_panel.add_column("Tier", style="yellow")
        rule_panel.add_column("Sublayer", style="dim")
        rule_panel.add_column("Distilled Invariant Rule", style="white")
        for r in rules:
            rule_panel.add_row(r["tier"], r["sublayer"], r["rule"])
        console.print(rule_panel)

    # Check DPO Training Pairs
    dpo_file = Path("training_pairs.jsonl")
    if dpo_file.exists():
        with open(dpo_file, "r", encoding="utf-8") as f:
            pair_count = sum(1 for _ in f)
        console.print(Panel(f"[bold green]Self-Training Dataset:[/bold green] {pair_count} DPO preference tuples exported to [bold]{dpo_file.name}[/bold] (Zero Human Labeling)"))


def cmd_fanout(args: argparse.Namespace) -> None:
    display_banner()
    console.print("[bold yellow][*] Demonstrating Parallel Worktree Fan-Out & Collision Arbitration...[/bold yellow]")

    engine = SimulationEngine(repo_path=args.repo)
    fanout_res = engine.run_parallel_fan_out_step()

    table = Table(title="Parallel Worktree Swarm Fan-Out", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Result", style="green")

    table.add_row("Worker A (AST Branch)", f"Commit: {str(fanout_res['worker_a_sha'])[:7]}")
    table.add_row("Worker B (EXEC Branch)", f"Commit: {str(fanout_res['worker_b_sha'])[:7]}")
    table.add_row("Integration Branch", fanout_res["integration_branch"])
    table.add_row("Collision Detected", "[bold red]YES[/bold red]" if fanout_res["collision_detected"] else "[green]NO (Clean Merge)[/green]")

    console.print(table)


def cmd_constraints(args: argparse.Namespace) -> None:
    display_banner()
    mgr = ConstraintManager()
    out_path = mgr.generate_and_save(human_prompt=args.prompt, filename=args.filename)
    chash = mgr.compute_contract_hash()

    console.print(Panel(
        f"[bold green]Synthesized Test Suite Saved:[/bold green] {out_path}\n"
        f"[bold cyan]Parent-Contract-Hash:[/bold cyan] {chash}\n\n"
        f"[dim]{out_path.read_text(encoding='utf-8')[:300]}...[/dim]",
        title="Human-in-the-Loop Constraint Synthesis",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zel Beta Stratified Forensic Swarm CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run command
    p_run = subparsers.add_parser("run", help="Run full simulation engine loop")
    p_run.add_argument("-g", "--generations", type=int, default=3, help="Number of evolutionary generations")
    p_run.add_argument("-r", "--repo", default="Simulation/Repo", help="Target Git repository directory")

    # fanout command
    p_fanout = subparsers.add_parser("fanout", help="Run parallel worktree fan-out demo")
    p_fanout.add_argument("-r", "--repo", default="Simulation/Repo", help="Target Git repository directory")

    # constraints command
    p_const = subparsers.add_parser("constraints", help="Synthesize human prompt into executable constraint test")
    p_const.add_argument("-p", "--prompt", required=True, help="Natural language requirement prompt")
    p_const.add_argument("-f", "--filename", default="test_constraints.py", help="Output test filename")

    args = parser.parse_args()

    if args.subcommand == "run":
        cmd_run(args)
    elif args.subcommand == "fanout":
        cmd_fanout(args)
    elif args.subcommand == "constraints":
        cmd_constraints(args)


if __name__ == "__main__":
    main()
