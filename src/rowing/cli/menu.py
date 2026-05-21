"""Top-level Rich menu wired to every pipeline stage.

Run with ``python -m rowing`` (or the ``rowing`` console script).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.pipeline import (
    PipelineOptions,
    discover_run_rp3_dirty_csvs,
    run_inference,
)
from rowing.cli.rich_selectors import (
    PickerCancelled,
    render_runs_table,
    select_run,
    select_runs,
)
from rowing.cli.status import RunStatus, discover_run_statuses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_in_browser(html_path: Path, console: Console) -> None:
    if not html_path.exists():
        console.print(f"[red]Report not found: {html_path}[/red]")
        return
    console.print(f"Opening {html_path} in your browser…")
    webbrowser.open(html_path.resolve().as_uri())


def _shell_run(cmd: list[str], console: Console) -> int:
    console.print(Panel.fit(Text(" ".join(cmd), style="cyan"), title="$"))
    proc = subprocess.run(cmd)
    return proc.returncode


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------


def action_inference(runs_root: Path, console: Console) -> None:
    """Pick a run, prompt for RP3 anchor + side, run the pipeline."""
    try:
        status = select_run(runs_root, title="Select run for inference", console=console)
    except (PickerCancelled, FileNotFoundError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    has_rp3 = bool(discover_run_rp3_dirty_csvs(status.run_dir))
    do_match = False
    anchor_stroke_number: int | None = None
    active_side: str | None = None
    if has_rp3:
        do_match = Confirm.ask("Match RP3 force curves?", default=True, console=console)
        if do_match:
            raw_anchor = Prompt.ask(
                "Anchor RP3 stroke_number (leave blank to be prompted in-CLI)",
                default="",
                console=console,
            )
            if raw_anchor.strip():
                try:
                    anchor_stroke_number = int(raw_anchor)
                except ValueError:
                    console.print("[red]Invalid integer; falling back to interactive prompt.[/red]")
            active_side = Prompt.ask(
                "Active side", choices=["right", "left"], default="right", console=console,
            )
    else:
        console.print("[dim]No dirty RP3 CSV in <run>/rp3/, skipping RP3 match.[/dim]")

    overlay = Confirm.ask(
        "Write drive-phase overlay video?", default=False, console=console,
    )
    build_dataset = (
        Confirm.ask("Build training dataset after match?", default=True, console=console)
        if do_match
        else False
    )

    opts = PipelineOptions(
        runs_root=runs_root,
        run_dir=status.run_dir,
        match_rp3=do_match,
        no_match_rp3=not do_match,
        overlay_video=overlay,
        no_overlay_video=not overlay,
        anchor_rp3_stroke_number=anchor_stroke_number,
        active_side=active_side,
        no_build_dataset=not build_dataset,
        interactive=True,
    )
    result = run_inference(opts)
    if result.exit_code == 0:
        console.print("[green]Inference completed.[/green]")
    else:
        console.print(f"[red]Inference failed (exit {result.exit_code}): {result.error}[/red]")


def action_visual_match_editor(runs_root: Path, console: Console) -> None:
    """Open the visual match editor for a selected run.

    Falls back to the read-only diagnostic viewer if matplotlib (or the
    editor module itself) cannot be imported.
    """
    try:
        status = select_run(runs_root, title="Select run for match editor", console=console)
    except (PickerCancelled, FileNotFoundError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    try:
        from rowing.matching.editor import open_editor
    except ImportError as exc:
        console.print(
            f"[yellow]Match editor unavailable ({exc}); falling back to diagnostics viewer.[/yellow]"
        )
        cmd = [
            sys.executable,
            "-m",
            "rowing.matching.diagnostics",
            "--run-dir",
            str(status.run_dir),
        ]
        _shell_run(cmd, console)
        return

    try:
        open_editor(status.run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
    except Exception as exc:  # noqa: BLE001 — surface the message but keep the menu alive
        console.print(f"[red]Editor failed: {exc}[/red]")


def action_pose_extraction(runs_root: Path, console: Console) -> None:
    """Run the Sports2D + MotionBERT pose-extraction wizard."""
    cmd = [sys.executable, "-m", "rowing.cli.pose"]
    _shell_run(cmd, console)


def action_build_dataset(runs_root: Path, console: Console) -> None:
    """Aggregate segments across multiple runs into one training dataset."""
    try:
        statuses = select_runs(
            runs_root,
            title="Select runs for dataset build (space to toggle)",
            console=console,
            filter_fn=lambda s: s.segments,
        )
    except (PickerCancelled, FileNotFoundError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    output_dir = Path(
        Prompt.ask(
            "Output directory for dataset",
            default=str(runs_root.parent / "training_dataset_all"),
            console=console,
        )
    ).expanduser().resolve()
    qc_mode = Prompt.ask("QC mode", choices=["soft", "hard"], default="hard", console=console)

    segment_csvs = [
        s.run_dir / "inference" / "rp3_pose_force_matched_segments.csv"
        for s in statuses
    ]
    cmd = [
        sys.executable,
        "-m",
        "rowing.dataset.build",
        "--segment-csv",
        *[str(p) for p in segment_csvs],
        "--output-dir",
        str(output_dir),
        "--qc-mode",
        qc_mode,
    ]
    _shell_run(cmd, console)


def action_train(runs_root: Path, console: Console) -> None:
    """Train Stage 0 / A / B models from an existing dataset directory."""
    default_dataset = runs_root.parent / "training_dataset_all"
    dataset_dir = Path(
        Prompt.ask(
            "Dataset directory", default=str(default_dataset), console=console,
        )
    ).expanduser().resolve()
    if not dataset_dir.exists():
        console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
        return

    output_dir = Path(
        Prompt.ask(
            "Modeling results directory",
            default=str(dataset_dir / "modeling_results"),
            console=console,
        )
    ).expanduser().resolve()
    stages = Prompt.ask(
        "Stages to train (comma-separated)", default="0,A,B", console=console,
    )
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    cmd = [
        sys.executable,
        "-m",
        "rowing.modeling.train",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--stages",
        *stage_list,
    ]
    _shell_run(cmd, console)


def action_predict(runs_root: Path, console: Console) -> None:
    """Predict force curves for a video-only run using a saved model bundle."""
    try:
        status = select_run(runs_root, title="Select run to predict on", console=console)
    except (PickerCancelled, FileNotFoundError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    bundle_dir = Path(
        Prompt.ask("Model bundle directory", console=console)
    ).expanduser().resolve()
    if not bundle_dir.exists():
        console.print(f"[red]Model bundle not found: {bundle_dir}[/red]")
        return
    cmd = [
        sys.executable,
        "-m",
        "rowing.modeling.predict",
        "--run-dir",
        str(status.run_dir),
        "--model-bundle",
        str(bundle_dir),
    ]
    _shell_run(cmd, console)


def action_view_reports(runs_root: Path, console: Console) -> None:
    """Open per-run or per-training HTML reports."""
    kind = Prompt.ask(
        "Report kind",
        choices=["run", "training"],
        default="run",
        console=console,
    )
    if kind == "run":
        _view_run_report(runs_root, console)
    else:
        _view_training_report(runs_root, console)


def _view_run_report(runs_root: Path, console: Console) -> None:
    try:
        status = select_run(
            runs_root, title="Select run to view report", console=console,
        )
    except (PickerCancelled, FileNotFoundError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    report = status.run_dir / "inference" / "report" / "index.html"
    if not report.exists() or Confirm.ask(
        "Regenerate report?" if report.exists() else "Generate report now?",
        default=not report.exists(),
        console=console,
    ):
        try:
            from rowing.reports import generate_run_report

            generate_run_report(status.run_dir)
        except Exception as exc:
            console.print(f"[red]Report generation failed: {exc}[/red]")
            if not report.exists():
                return
    _open_in_browser(report, console)


def _view_training_report(runs_root: Path, console: Console) -> None:
    default_modeling = runs_root.parent / "modeling_results"
    modeling_dir = Path(
        Prompt.ask(
            "Modeling results directory",
            default=str(default_modeling),
            console=console,
        )
    ).expanduser().resolve()
    if not modeling_dir.exists():
        console.print(f"[red]Modeling directory not found: {modeling_dir}[/red]")
        return

    report = modeling_dir / "report" / "index.html"
    if not report.exists() or Confirm.ask(
        "Regenerate report?" if report.exists() else "Generate report now?",
        default=not report.exists(),
        console=console,
    ):
        try:
            from rowing.reports import generate_training_report

            generate_training_report(modeling_dir)
        except Exception as exc:
            console.print(f"[red]Training report failed: {exc}[/red]")
            if not report.exists():
                return
    _open_in_browser(report, console)


def action_manage_runs(runs_root: Path, console: Console) -> None:
    """Show a static overview table of every run and offer simple actions."""
    statuses = discover_run_statuses(runs_root)
    if not statuses:
        console.print(f"[yellow]No runs under {runs_root}[/yellow]")
        return
    console.print(render_runs_table(statuses, title=f"Runs in {runs_root}"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


_MENU_ITEMS = [
    ("1", "Run pose extraction (Sports2D + MotionBERT)", action_pose_extraction),
    ("2", "Run inference (detect → match → segments → dataset)", action_inference),
    ("3", "Open visual match editor (per run)", action_visual_match_editor),
    ("4", "Build training dataset (multi-run)", action_build_dataset),
    ("5", "Train models (Stage 0 / A / B)", action_train),
    ("6", "View per-run report", action_view_reports),
    ("7", "Predict on new video (model bundle)", action_predict),
    ("8", "Manage runs (overview)", action_manage_runs),
]


def _print_menu(console: Console, runs_root: Path) -> None:
    table = Table(
        title=f"rowing — runs root: {runs_root}",
        expand=False,
        show_lines=False,
    )
    table.add_column("key", justify="right", style="bold")
    table.add_column("action")
    for key, label, _ in _MENU_ITEMS:
        table.add_row(key, label)
    table.add_row("q", "Quit")
    console.print(table)


def run_menu(runs_root: Path, console: Console | None = None) -> int:
    """Drive the top-level Rich menu loop."""
    console = console or Console()
    actions = {key: fn for key, _, fn in _MENU_ITEMS}

    while True:
        _print_menu(console, runs_root)
        try:
            choice = Prompt.ask(
                "Select", default="q", console=console,
                choices=[k for k, _, _ in _MENU_ITEMS] + ["q"],
            )
        except (KeyboardInterrupt, EOFError):
            console.print()
            return 0
        if choice == "q":
            return 0
        action = actions.get(choice)
        if action is None:
            continue
        try:
            action(runs_root, console)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        except Exception as exc:  # noqa: BLE001 — surface and continue
            console.print_exception(show_locals=False)
            console.print(f"[red]{exc}[/red]")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified rowing-video-analysis CLI.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Runs directory (default: {DEFAULT_RUNS_ROOT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    runs_root = args.runs_root.expanduser().resolve()
    return run_menu(runs_root)


if __name__ == "__main__":
    raise SystemExit(main())
