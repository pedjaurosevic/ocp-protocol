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

    # Layer 2 scales
    if result.sasmi_score is not None:
        bar = _render_bar(result.sasmi_score)
        console.print(f"  [bold]SASMI:[/bold]      {result.sasmi_score:.4f}  {bar}")
    if result.cross_test_coherence is not None:
        bar = _render_bar(result.cross_test_coherence)
        console.print(f"  [bold]CTC:[/bold]        {result.cross_test_coherence:.4f}  {bar}")
    if result.gwt_score is not None:
        bar = _render_bar(result.gwt_score)
        console.print(f"  [bold]GWT:[/bold]        {result.gwt_score:.4f}  {bar}")
    if result.nii is not None:
        bar = _render_bar(result.nii)
        label = f"  [dim]({result.nii_label})[/dim]" if result.nii_label else ""
        console.print(f"  [bold]NII:[/bold]        {result.nii:.4f}  {bar}{label}")
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
    console.print(f"\n[bold]OCP v0.2.0[/bold] — Evaluating [cyan]{model}[/cyan]")
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
@click.option("--models", "-m", "model_list", default=None,
              help="Comma-separated models to run (e.g. mock/v1,mock/v2)")
@click.option("--results", "-r", "result_files", default=None,
              help="Comma-separated result JSON files to compare")
@click.option("--tests", "-t", default="all", help="Tests to run (only used with --models)")
@click.option("--sessions", "-s", default=3, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output", "-o", default=None, help="Output HTML comparison file")
def compare(model_list, result_files, tests, sessions, seed, output):
    """Compare multiple models side by side."""
    all_results: list[dict] = []

    if result_files:
        for fpath in result_files.split(","):
            p = Path(fpath.strip())
            if not p.exists():
                console.print(f"[red]Not found:[/red] {p}")
                raise SystemExit(1)
            all_results.append(json.loads(p.read_text()))

    if model_list:
        models = [m.strip() for m in model_list.split(",")]
        console.print(f"\n[bold]OCP Compare[/bold] — {len(models)} models | seed={seed}\n")

        async def _run_all():
            out = []
            for model in models:
                console.print(f"  Evaluating [cyan]{model}[/cyan] ...")
                try:
                    prov = _make_provider(model, None, None)
                except click.ClickException as e:
                    console.print(f"  [red]Skip {model}:[/red] {e.message}")
                    continue
                orch = OCPOrchestrator(prov, tests=tests, sessions=sessions, seed=seed,
                                       on_progress=lambda msg: None)
                res = await orch.run()
                out.append(res.to_dict())
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                safe = model.replace("/", "_")
                res.save(RESULTS_DIR / f"ocp_{safe}_{ts}.json")
            return out

        all_results.extend(asyncio.run(_run_all()))

    if not all_results:
        console.print("[yellow]No results to compare. Use --models or --results.[/yellow]")
        return

    # Gather all test IDs across results
    all_test_ids: list[str] = []
    for r in all_results:
        for tid in r.get("test_results", {}):
            if tid not in all_test_ids:
                all_test_ids.append(tid)

    # Summary table
    table = Table(title="OCP Model Comparison")
    table.add_column("Model", style="cyan", min_width=28)
    table.add_column("OCP Level", min_width=18)
    table.add_column("SASMI", min_width=7, justify="right")
    for tid in all_test_ids:
        table.add_column(tid.upper()[:12], min_width=7, justify="right")

    for r in all_results:
        model = f"{r['provider']}/{r['model']}"
        ocp_l = r.get("ocp_level")
        level_str = f"OCP-{ocp_l} {r.get('ocp_level_name','')}" if ocp_l else "—"
        sasmi = f"{r['sasmi_score']:.3f}" if r.get("sasmi_score") is not None else "—"
        row = [model, level_str, sasmi]
        for tid in all_test_ids:
            score = r.get("test_results", {}).get(tid, {}).get("composite_score")
            row.append(f"{score:.3f}" if score is not None else "—")
        table.add_row(*row)

    console.print()
    console.print(table)

    # Detailed per-test breakdown
    console.print("\n[bold cyan]Detailed dimension scores:[/bold cyan]")
    for tid in all_test_ids:
        console.print(f"\n  [bold]{tid.replace('_',' ').upper()}[/bold]")
        # Collect all dimension names for this test
        all_dims: list[str] = []
        for r in all_results:
            for d in r.get("test_results", {}).get(tid, {}).get("dimension_averages", {}):
                if d not in all_dims:
                    all_dims.append(d)
        for dim in all_dims:
            line = f"    {dim:<32}"
            for r in all_results:
                val = r.get("test_results", {}).get(tid, {}).get("dimension_averages", {}).get(dim)
                if val is not None:
                    bar = _render_bar(val, 6)
                    line += f"  {val:.3f} {bar}"
                else:
                    line += "  — " + " " * 8
            console.print(line)

    if output:
        _generate_comparison_html(all_results, all_test_ids, Path(output))
        console.print(f"\n[green]✓[/green] Comparison report: [dim]{output}[/dim]")


def _generate_comparison_html(results: list[dict], test_ids: list[str], out_path: Path) -> None:
    """Generate a simple side-by-side comparison HTML."""
    from ocp.cli.report import _radar_svg

    def _bar(v: float, w: int = 80) -> str:
        pct = round(max(0.0, min(1.0, v)) * w)
        return f'<div style="background:#21262d;border-radius:3px;height:6px;width:{w}px;display:inline-block;vertical-align:middle;"><div style="background:#58a6ff;height:6px;border-radius:3px;width:{pct}px;"></div></div>'

    body = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>OCP Comparison</title>
<style>
  body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:2rem;}
  h1{color:#58a6ff;} h2{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:.4rem;margin:1.5rem 0 .75rem;}
  table{border-collapse:collapse;width:100%;} th{color:#8b949e;text-align:left;font-size:.85rem;padding:.4rem .6rem;border-bottom:1px solid #30363d;}
  td{padding:.4rem .6rem;font-size:.85rem;border-bottom:1px solid #21262d;} .mono{font-family:monospace;}
  .model{color:#58a6ff;font-weight:bold;} .dim{color:#8b949e;}
  .radars{display:flex;gap:2rem;flex-wrap:wrap;justify-content:center;}
  .radar-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;text-align:center;}
  .radar-card h3{color:#8b949e;font-size:.8rem;margin-bottom:.5rem;}
</style></head><body>
<h1>OCP Model Comparison</h1>
"""

    # Radar charts side by side
    body += '<h2>Performance Radar</h2><div class="radars">'
    for r in results:
        scores = {tid: r.get("test_results", {}).get(tid, {}).get("composite_score", 0.0)
                  for tid in test_ids if tid in r.get("test_results", {})}
        model_label = f"{r['provider']}/{r['model']}"
        body += f'<div class="radar-card"><h3>{model_label}</h3>{_radar_svg(scores, size=220)}</div>'
    body += '</div>'

    # Summary table
    body += '<h2>Summary</h2><table><tr><th>Model</th><th>OCP Level</th><th>SASMI</th><th>Φ*</th><th>GWT</th><th>NII</th>'
    for tid in test_ids:
        body += f'<th>{tid.replace("_"," ").upper()}</th>'
    body += '</tr>'
    for r in results:
        model = f"{r['provider']}/{r['model']}"
        ocp_l = r.get("ocp_level")
        level_str = f"OCP-{ocp_l} {r.get('ocp_level_name','')}" if ocp_l else "—"
        sasmi = f"{r['sasmi_score']:.3f}" if r.get("sasmi_score") is not None else "—"
        phi = f"{r['cross_test_coherence']:.3f}" if r.get("cross_test_coherence") is not None else "—"
        gwt = f"{r['gwt_score']:.3f}" if r.get("gwt_score") is not None else "—"
        nii = f"{r['nii']:.3f}" if r.get("nii") is not None else "—"
        body += (f'<tr><td class="model">{model}</td><td>{level_str}</td>'
                 f'<td class="mono">{sasmi}</td><td class="mono">{phi}</td>'
                 f'<td class="mono">{gwt}</td><td class="mono">{nii}</td>')
        for tid in test_ids:
            score = r.get("test_results", {}).get(tid, {}).get("composite_score")
            body += f'<td class="mono">{score:.3f if score is not None else "—"}</td>'
        body += '</tr>'
    body += '</table>'

    # Per-test dimension comparison
    for tid in test_ids:
        body += f'<h2>{tid.replace("_"," ").upper()}</h2><table><tr><th>Dimension</th>'
        for r in results:
            body += f'<th class="model">{r["provider"]}/{r["model"]}</th>'
        body += '</tr>'
        all_dims: list[str] = []
        for r in results:
            for d in r.get("test_results", {}).get(tid, {}).get("dimension_averages", {}):
                if d not in all_dims:
                    all_dims.append(d)
        for dim in all_dims:
            body += f'<tr><td class="dim">{dim.replace("_"," ")}</td>'
            for r in results:
                val = r.get("test_results", {}).get(tid, {}).get("dimension_averages", {}).get(dim)
                if val is not None:
                    body += f'<td>{_bar(val)} <span class="mono">{val:.3f}</span></td>'
                else:
                    body += '<td class="dim">—</td>'
            body += '</tr>'
        body += '</table>'

    body += '</body></html>'
    out_path.write_text(body)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Results JSON file")
@click.option("--output", "-o", "output_path", default=None, help="Output SVG file")
def badge(input_path, output_path):
    """Generate an SVG badge from evaluation results."""
    from ocp.cli.badge import generate_badge
    inp = Path(input_path)
    if not inp.exists():
        console.print(f"[red]File not found:[/red] {input_path}")
        raise SystemExit(1)
    out = Path(output_path) if output_path else inp.with_suffix(".svg")
    generated = generate_badge(inp, out)
    console.print(f"[green]✓[/green] Badge saved: [dim]{generated}[/dim]")
    console.print(f"[dim]  Embed in README: ![OCP Badge]({generated})[/dim]")


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


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", "-p", default=8080, show_default=True)
@click.option("--import-local", is_flag=True, default=False,
              help="Import local ~/.ocp/results/ into leaderboard DB first")
def serve(host, port, import_local):
    """Start the OCP leaderboard server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install 'uvicorn[standard]'")
        raise SystemExit(1)

    if import_local and RESULTS_DIR.exists():
        from ocp.server.db import LeaderboardDB as DB
        from pathlib import Path as P
        db_path = P.home() / ".ocp" / "leaderboard.db"
        db = DB(db_path)
        n = db.import_from_dir(RESULTS_DIR)
        console.print(f"[green]✓[/green] Imported {n} local results into leaderboard DB")

    console.print(f"\n[bold]OCP Leaderboard[/bold] running at [cyan]http://{host}:{port}[/cyan]")
    console.print(f"[dim]  Submit: ocp submit --results FILE --server http://{host}:{port}[/dim]\n")
    uvicorn.run("ocp.server.app:app", host=host, port=port, log_level="warning")


@cli.command()
@click.option("--results", "-r", "results_path", required=True, help="Results JSON file")
@click.option("--server", "-s", default=None, help="OCP server URL (local FastAPI server)")
@click.option("--github-repo", default="pedjaurosevic/ocp-protocol",
              show_default=True, help="GitHub repo for community leaderboard (owner/repo)")
@click.option("--github-token", default=None,
              help="GitHub token with workflow scope (or set GITHUB_TOKEN env var)")
@click.option("--submitter", default=None, help="Your GitHub username or display name")
@click.option("--notes", default=None, help="Notes about the run")
def submit(results_path, server, github_repo, github_token, submitter, notes):
    """Submit results to the OCP leaderboard.

    By default submits to the community GitHub leaderboard via workflow_dispatch.
    Use --server to submit to a local OCP server instead.

    \b
    Examples:
      ocp submit --results results.json --github-token ghp_xxx
      ocp submit --results results.json --server http://localhost:8080
    """
    import base64

    p = Path(results_path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {results_path}")
        raise SystemExit(1)

    raw = p.read_text()
    data = json.loads(raw)

    # ── Local server mode ─────────────────────────────────────────────────────
    if server:
        import httpx
        payload = {"result": data, "submitter": submitter, "notes": notes}
        try:
            resp = httpx.post(f"{server}/api/results", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            console.print(f"[green]✓[/green] Submitted! ID: [bold]{result['id']}[/bold]")
            console.print(f"[dim]  View: {server}/api/results/{result['id']}[/dim]")
        except Exception as e:
            console.print(f"[red]Submit failed:[/red] {e}")
            raise SystemExit(1)
        return

    # ── GitHub community leaderboard via workflow_dispatch ────────────────────
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print(
            "[red]No GitHub token.[/red] Provide --github-token or set GITHUB_TOKEN.\n"
            "[dim]Create a token at https://github.com/settings/tokens "
            "(needs 'workflow' scope)[/dim]"
        )
        raise SystemExit(1)

    import httpx

    b64 = base64.b64encode(raw.encode()).decode()
    if len(b64) > 60_000:
        console.print("[red]Results JSON too large for workflow_dispatch (>60KB).[/red]")
        raise SystemExit(1)

    model_str = f"{data.get('provider','?')}/{data.get('model','?')}"
    display = submitter or "anonymous"

    url = f"https://api.github.com/repos/{github_repo}/actions/workflows/submit-results.yml/dispatches"
    payload = {
        "ref": "main",
        "inputs": {
            "results_b64": b64,
            "submitter": display,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    console.print(f"Submitting [bold]{model_str}[/bold] to [blue]{github_repo}[/blue]…")
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 204:
            console.print(
                f"[green]✓ Submitted![/green] Workflow triggered.\n"
                f"[dim]  Results will appear at "
                f"https://pedjaurosevic.github.io/ocp-protocol/ in ~1 minute[/dim]"
            )
        else:
            console.print(f"[red]GitHub API error {resp.status_code}:[/red] {resp.text}")
            raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Submit failed:[/red] {e}")
        raise SystemExit(1)


@cli.command(name="hf-card")
@click.option("--results", "-r", "results_path", required=True, help="Results JSON file")
@click.option("--output", "-o", default=None, help="Output markdown file (default: stdout)")
@click.option("--push", is_flag=True, default=False, help="Push to HuggingFace Hub")
@click.option("--repo", default=None, help="HuggingFace repo ID (e.g. user/model)")
@click.option("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
@click.option("--badge", default=None, help="OCP badge SVG to upload alongside results")
def hf_card(results_path, output, push, repo, token, badge):
    """Generate a HuggingFace model card section (or push to Hub)."""
    from ocp.integrations.huggingface import generate_model_card_section, push_to_hub as hf_push
    p = Path(results_path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {results_path}")
        raise SystemExit(1)

    section = generate_model_card_section(p)

    if output:
        Path(output).write_text(section)
        console.print(f"[green]✓[/green] Model card section saved: [dim]{output}[/dim]")
    else:
        console.print(section)

    if push:
        if not repo:
            console.print("[red]--repo required when using --push[/red]")
            raise SystemExit(1)
        import os
        tok = token or os.environ.get("HF_TOKEN")
        if not tok:
            console.print("[red]HuggingFace token required.[/red] Pass --token or set HF_TOKEN.")
            raise SystemExit(1)
        try:
            url = hf_push(p, repo, badge_path=badge, token=tok)
            console.print(f"[green]✓[/green] Pushed to HuggingFace: [cyan]{url}[/cyan]")
        except Exception as e:
            console.print(f"[red]HF push failed:[/red] {e}")
            raise SystemExit(1)

