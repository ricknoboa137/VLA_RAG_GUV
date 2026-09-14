"""``carma`` command-line interface.

Flags here control *where things go*, never what the science does. Anything
that changes behaviour belongs in a config file; see ``AGENTS.md`` section 2,
rule 4.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import structlog
import typer
import yaml

from carma.cli.manifest import build_manifest, read_manifest, write_manifest
from carma.config import load
from carma.runtime import run_experiment
from carma.types import CarmaError

app = typer.Typer(add_completion=False, help="Cost-aware retrieval memory arbitration.")


def _configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=(run_dir / "decisions.jsonl").open("a")),
    )


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Experiment config YAML.")],
    seed: Annotated[int | None, typer.Option(help="Override the config seed.")] = None,
    out: Annotated[Path, typer.Option(help="Root directory for run outputs.")] = Path("runs"),
) -> None:
    """Run one experiment and write its manifest, metrics and decision log."""
    try:
        cfg = load(config)
    except CarmaError as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if seed is not None:
        cfg = replace(cfg, seed=seed)

    run_id = f"{cfg.name}-{cfg.condition}-{uuid.uuid4().hex[:8]}"
    run_dir = out / run_id
    _configure_logging(run_dir)

    manifest = build_manifest(cfg, run_id)
    write_manifest(manifest, run_dir)
    if manifest.git_dirty:
        typer.secho(
            "warning: working tree is dirty; this run is not reproducible",
            fg=typer.colors.YELLOW,
            err=True,
        )

    try:
        result = run_experiment(cfg, run_id=run_id)
    except CarmaError as exc:
        typer.secho(f"run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    (run_dir / "metrics.yaml").write_text(yaml.safe_dump(result.metrics.to_dict(), sort_keys=True))
    typer.echo(json.dumps(result.metrics.to_dict(), indent=2, sort_keys=True))
    typer.secho(f"run written to {run_dir}", fg=typer.colors.GREEN)


@app.command()
def report(
    runs: Annotated[Path, typer.Argument(help="Directory holding run subdirectories.")],
) -> None:
    """Aggregate runs, refusing any set whose config hashes disagree."""
    dirs = sorted(d for d in runs.iterdir() if d.is_dir()) if runs.exists() else []
    if not dirs:
        typer.secho(f"no runs under {runs}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    hashes: dict[str, list[str]] = {}
    for d in dirs:
        try:
            manifest = read_manifest(d)
        except (FileNotFoundError, ValueError) as exc:
            typer.secho(f"skipping {d.name}: {exc}", fg=typer.colors.YELLOW, err=True)
            continue
        hashes.setdefault(str(manifest["config_hash"]), []).append(d.name)

    if len(hashes) > 1:
        typer.secho("refusing to aggregate: config hashes disagree", fg=typer.colors.RED, err=True)
        for h, names in sorted(hashes.items()):
            typer.echo(f"  {h}: {', '.join(names)}", err=True)
        raise typer.Exit(code=3)

    for h, names in hashes.items():
        typer.echo(f"config_hash {h} over {len(names)} run(s)")
        for name in names:
            metrics_path = runs / name / "metrics.yaml"
            if metrics_path.exists():
                typer.echo(f"  {name}: {metrics_path.read_text().strip()}")


@app.command()
def components() -> None:
    """List every registered component key, for writing configs against."""
    from carma.arbiter.cost_model import COST_MODELS
    from carma.arbiter.policies import ARBITERS
    from carma.backbone.registry import BACKBONES
    from carma.memory.embedders.registry import EMBEDDERS
    from carma.memory.hygiene import HYGIENE
    from carma.memory.retriever import RETRIEVERS
    from carma.memory.stores.registry import STORES
    from carma.operator.registry import OPERATORS
    from carma.sim.registry import SIMS

    for label, reg in (
        ("backbone", BACKBONES),
        ("embedder", EMBEDDERS),
        ("store", STORES),
        ("retriever", RETRIEVERS),
        ("hygiene", HYGIENE),
        ("cost_model", COST_MODELS),
        ("arbiter", ARBITERS),
        ("operator", OPERATORS),
        ("sim", SIMS),
    ):
        typer.echo(f"{label}: {', '.join(reg.keys())}")


def main() -> int:
    """Console-script entry point."""
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
