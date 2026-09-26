"""CLI entry point for auto-eval-platform."""
from __future__ import annotations
import typer
from rich.console import Console

app = typer.Typer(help="auto-eval-platform CLI")
console = Console()

@app.command()
def evaluate(
    split: str = typer.Option("test", help="Split to evaluate: train/val/test"),
    version: str = typer.Option(None, help="Golden set version (default: latest)"),
    config: str = typer.Option("configs/config.yaml", help="Config file path"),
):
    """Run evaluation on specified split."""
    console.print(f"[cyan]Evaluating split=[{split}] version=[{version or 'latest'}][/cyan]")
    # TODO: Load config, golden set, run harness, print report
    console.print("[green]✓ Evaluation complete[/green]")

@app.command()
def golden_set(
    action: str = typer.Argument(..., help="Action: create/refresh/verify/curate"),
    version: str = typer.Option(None, help="Version"),
    source: str = typer.Option(None, help="Source data path"),
):
    """Manage golden sets via GoldenSetAgent."""
    console.print(f"[cyan]Golden Set Agent: {action}[/cyan]")

@app.command()
def red_team(
    vectors: str = typer.Option("all", help="Attack vectors (comma-separated)"),
    schedule: str = typer.Option("nightly", help="Schedule: nightly/now"),
):
    """Run Red Team Agent suite."""
    console.print(f"[cyan]Red Team Agent: {vectors} [{schedule}][/cyan]")

@app.command()
def drift(
    reference: str = typer.Option(..., help="Reference embeddings path"),
    current: str = typer.Option(..., help="Current embeddings path"),
):
    """Run Drift Agent check."""
    console.print(f"[cyan]Drift Agent: checking PSI/KL[/cyan]")

@app.command()
def compliance(
    regulation: str = typer.Option(..., help="Regulation file path"),
    output: str = typer.Option("tests/compliance", help="Output directory"),
):
    """Generate compliance test suite from regulation."""
    console.print(f"[cyan]Compliance Agent: ingesting {regulation}[/cyan]")

@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Start FastAPI server."""
    import uvicorn
    console.print(f"[cyan]Starting API server on {host}:{port}[/cyan]")
    uvicorn.run("auto_eval.api.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    app()
