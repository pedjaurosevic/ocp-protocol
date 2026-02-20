"""
OCP CLI — ocp evaluate, ocp tests, ocp leaderboard
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ocp.engine.orchestrator import OCPOrchestrator, AVAILABLE_TESTS
from ocp.engine.session import EvaluationResult

console = Console()

RESULTS_DIR = Path.home() / ".ocp" / "results"


@click.group()
@click.version_option("0.1.0", prog_name="ocp")
def cli():
    """OCP — Open Consciousness Protocol benchmark tool."""
    pass


def _make_provider(model: str, base_url: str | None, api_key: str | None):
    """Parse model string like 'groq/llama-3.3-70b' and return provider."""
    if "/" in model:
        provider_name, model_name = model.split("/", 1)
    else:
        provider_name, model_name = "groq", model

    if provider_name == "mock":
        from ocp.providers.mock import MockProvider
        return MockProvider(model=model_name)
    elif provider_name == "groq":
        from ocp.providers.groq import GroqProvider
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise click.ClickException("GROQ_API_KEY not set. Export it or pass --api-key.")
        return GroqProvider(model=model_name, api_key=key)
    elif provider_name == "ollama":
        from ocp.providers.ollama import OllamaProvider
        return OllamaProvider(model=model_name, base_url=base_url)
    elif provider_name in ("custom", "openai", "local"):
        from ocp.providers.openai_compat import OpenAICompatProvider
        url = base_url or os.environ.get("OCP_BASE_URL", "http://localhost:8080/v1")
        return OpenAICompatProvider(model=model_name, base_url=url, api_key=api_key,
                                    provider_name=provider_name)
    else:
        raise click.ClickException(
            f"Unknown provider: '{provider_name}'. Available: groq, ollama, mock, custom"
        )


def _render_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_results(result: EvaluationResult) -> None:
    console.print()
    console.print(Panel(
        f"[bold cyan]OCP Evaluation Results[/bold cyan]\n[dim]Protocol v{result.protocol_version}[/dim]",
        expand=False,
    ))
    console.print(f"  Model:    [bold]{result.provider}/{result.model}[/bold]")
    console.print(f"  Seed:     {result.seed}")
    console.print()

    if result.ocp_level is not None:
        level_colors = {1: "dim", 2: "yellow", 3: "green", 4: "cyan", 5: "bold magenta"}
        color = level_colors.get(result.ocp_level, "white")
        console.print(f"  [bold]OCP Level:[/bold]  [{color}]OCP-{result.ocp_level} — {result.ocp_level_name}[/{color}]")

    if result.sasmi_score is not None:
        bar = _render_bar(result.sasmi_score)
        console.print(f"  [bold]SASMI:[/bold]      {result.sasmi_score:.2f}  {bar}")
    console.print()

    for test_id, test_result in result.test_results.items():
        console.print(f"  [bold underline]{test_id}[/bold underline]  composite: {test_result.composite_score:.3f}")
        for dim, avg in test_result.dimension_averages.items():
            bar = _render_bar(avg, width=8)
            console.print(f"    ├─ {dim:<30} {avg:.3f}  {bar}")
        console.print()


@cli.command()
@click.option("--model", "-m", required=True,
              help="Model to evaluate (e.g. groq/llama-3.3-70b-versatile, mock/v1)")
@click.option("--tests", "-t", default="all", help="Tests to run (comma-separated or 'all')")
@click.option("--sessions", "-s", default=5, show_default=True, help="Sessions per test")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
@click.option("--output", "-o", default=None, help="Output JSON file path")
@click.option("--api-key", default=None, help="API key (overrides env var)")
@click.option("--base-url", default=None, help="Custom API base URL")
def evaluate(model, tests, sessions, seed, output, api_key, base_url):
    """Run an OCP evaluation on a model."""
    console.print(f"\n[bold]OCP v0.1.0[/bold] — Evaluating [cyan]{model}[/cyan]")
    console.print(f"[dim]Tests: {tests} | Sessions: {sessions} | Seed: {seed}[/dim]\n")

    try:
        provider = _make_provider(model, base_url, api_key)
    except click.ClickException as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise SystemExit(1)

    def on_progress(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    async def _run():
        orchestrator = OCPOrchestrator(
            provider=provider, tests=tests, sessions=sessions,
            seed=seed, on_progress=on_progress,
        )
        return await orchestrator.run()

    try:
        result = asyncio.run(_run())
    except Exception as e:
        console.print(f"\n[red]Evaluation failed:[/red] {e}")
        raise SystemExit(1)

    _print_results(result)

    if output:
        out_path = Path(output)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_")
        out_path = RESULTS_DIR / f"ocp_{safe_model}_{ts}.json"

    saved = result.save(out_path)
    console.print(f"[green]✓[/green] Results saved: [dim]{saved}[/dim]")
    console.print(f"[dim]  ocp leaderboard  ← to see all results[/dim]\n")


@cli.command(name="tests")
@click.argument("subcommand", default="list", required=False)
@click.argument("test_id", default=None, required=False)
def tests_cmd(subcommand, test_id):
    """List available tests or show test details."""
    if subcommand == "list":
        table = Table(title="Available OCP Tests")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status", style="green")
        for tid, cls in AVAILABLE_TESTS.items():
            table.add_row(tid, getattr(cls, "test_name", tid), "✓ available")
        console.print(table)
    elif subcommand == "info" and test_id:
        cls = AVAILABLE_TESTS.get(test_id)
        if not cls:
            console.print(f"[red]Unknown test:[/red] {test_id}")
            raise SystemExit(1)
        console.print(Panel(f"[bold]{cls.test_name}[/bold]\n\n{cls.description}", title=f"[cyan]{test_id}[/cyan]"))
    else:
        console.print("Usage: ocp tests list | ocp tests info <test_id>")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Results JSON file")
@click.option("--output", "-o", "output_path", default=None, help="Output HTML file (default: same name, .html)")
def report(input_path, output_path):
    """Generate an HTML report from evaluation results."""
    from ocp.cli.report import generate_report
    inp = Path(input_path)
    if not inp.exists():
        console.print(f"[red]File not found:[/red] {input_path}")
        raise SystemExit(1)
    out = Path(output_path) if output_path else inp.with_suffix(".html")
    generated = generate_report(inp, out)
    console.print(f"[green]✓[/green] Report saved: [dim]{generated}[/dim]")
    console.print(f"[dim]  Open in browser: xdg-open {generated}[/dim]")


@cli.command()
def leaderboard():
    """Show local evaluation leaderboard."""
    if not RESULTS_DIR.exists():
        console.print("[dim]No results yet. Run: ocp evaluate --model mock/v1[/dim]")
        return

    results = []
    for f in sorted(RESULTS_DIR.glob("ocp_*.json"), reverse=True):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            continue

    if not results:
        console.print("[dim]No results yet.[/dim]")
        return

    table = Table(title="OCP Local Leaderboard")
    table.add_column("Model", style="cyan")
    table.add_column("OCP Level")
    table.add_column("SASMI")
    table.add_column("MCA")
    table.add_column("Sessions")
    table.add_column("Date")

    for r in results:
        model = f"{r['provider']}/{r['model']}"
        level = f"OCP-{r['ocp_level']} {r.get('ocp_level_name','')}" if r.get("ocp_level") else "—"
        sasmi = f"{r['sasmi_score']:.3f}" if r.get("sasmi_score") is not None else "—"
        mca_score = r.get("test_results", {}).get("meta_cognition", {}).get("composite_score")
        mca = f"{mca_score:.3f}" if mca_score is not None else "—"
        sessions = str(r.get("config", {}).get("sessions", "?"))
        date = time.strftime("%Y-%m-%d", time.localtime(r.get("timestamp", 0)))
        table.add_row(model, level, sasmi, mca, sessions, date)

    console.print(table)

