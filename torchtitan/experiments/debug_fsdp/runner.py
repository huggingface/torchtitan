import toml
from argparse import ArgumentParser
from pathlib import Path
import re
import os
import subprocess
from enum import Enum
from jinja2 import Template
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "debugmodel_moe": {
        "torchtitan_config": "configs/torchtitan_debug.toml",
        "hf_fsdp_config": "configs/hf_qwen3_moe_fsdp_debug.toml",
        "torchtitan_flavor": "debugmodel_moe",
        "n_proc_per_node": 2,
    },
    "30B-A3B": {
        "torchtitan_config": "configs/torchtitan_30b_a3b.toml",
        "hf_fsdp_config": "configs/hf_qwen3_30b_a3b_fsdp.toml",
        "torchtitan_flavor": "30B-A3B",
        "n_proc_per_node": 8,
    },
    "llama3_8b": {
        "torchtitan_config": "configs/torchtitan_llama3_8b.toml",
        "hf_fsdp_config": "configs/hf_llama3_8b_fsdp.toml",
        "torchtitan_flavor": "8B",
        "n_proc_per_node": 8,
    },
}

# ---------------------------------------------------------------------------
# FSDP scenario definitions
# ---------------------------------------------------------------------------

ALL_FSDP_SCENARIOS = {
    "fsdp": {
        "enable_cpu_offload": False,
        "mixed_precision_param": None,
        "mixed_precision_reduce": None,
    },
    "fsdp_mixed_precision": {
        "enable_cpu_offload": False,
        "mixed_precision_param": "bfloat16",
        "mixed_precision_reduce": "float32",
    },
    "fsdp_cpu_offload": {
        "enable_cpu_offload": True,
        "mixed_precision_param": None,
        "mixed_precision_reduce": None,
    },
    "fsdp_cpu_offload_mixed_precision": {
        "enable_cpu_offload": True,
        "mixed_precision_param": "bfloat16",
        "mixed_precision_reduce": "float32",
    },
}


class LogLevel(Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TEST_PASS = "TEST_PASS"
    TEST_FAIL = "TEST_FAIL"


def log_message(level: LogLevel, message: str, indent: int = 0, dim: bool = False) -> None:
    style_map = {
        LogLevel.INFO: "blue",
        LogLevel.SUCCESS: "green",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "bold red",
        LogLevel.TEST_PASS: "green",
        LogLevel.TEST_FAIL: "bold red",
    }
    prefix_map = {
        LogLevel.INFO: "[INFO]",
        LogLevel.SUCCESS: "[SUCCESS]",
        LogLevel.WARNING: "[WARNING]",
        LogLevel.ERROR: "[ERROR]",
        LogLevel.TEST_PASS: "✅ TEST PASS",
        LogLevel.TEST_FAIL: "❌ TEST FAIL",
    }

    style = style_map[level]
    prefix = prefix_map[level]
    indent_str = "  " * (indent - 1) + "└─ " if indent > 0 else ""
    output = f"{indent_str}[{style}]{prefix}[/] {message}"

    if dim:
        console.print(f"[dim]{output}[/dim]")
    else:
        console.print(output)


# ---------------------------------------------------------------------------
# Config / SLURM generation
# ---------------------------------------------------------------------------

def _create_slurm_script(
    script_path: Path,
    job_name: str,
    config_path: str,
    n_proc_per_node: int = 2,
    is_seed: bool = False,
    seed_checkpoint_path: str = None,
    use_hf_fsdp: bool = False,
    enable_cpu_offload: bool = False,
    mixed_precision_param: str = None,
    mixed_precision_reduce: str = None,
    model_flavor: str = None,
    qos: str = "normal",
):
    template_path = SCRIPT_DIR / "template.slurm"
    with open(template_path, "r") as f:
        slurm_template = Template(f.read())

    context = {
        "name": job_name,
        "root_path": script_path.parent,
        "config_path": config_path,
        "n_proc_per_node": n_proc_per_node,
        "qos": qos,
        "is_seed": is_seed,
        "seed_checkpoint_path": seed_checkpoint_path,
        "use_hf_fsdp": use_hf_fsdp,
        "enable_cpu_offload": enable_cpu_offload,
        "mixed_precision_param": mixed_precision_param,
        "mixed_precision_reduce": mixed_precision_reduce,
        "model_flavor": model_flavor,
    }

    with open(script_path, "w") as f:
        f.write(slurm_template.render(context))
    print(f"Created SLURM script: {script_path}")


def create_configs(out_dir: str, model: str = "debugmodel_moe",
                   scenarios: list[str] = None, qos: str = "normal"):
    """
    Create output directory structure:

    out_dir/
        seed_checkpoint/
            seed.slurm
        <scenario>/
            torchtitan/
                run.slurm
            hf_fsdp/
                run.slurm
        ...
    """
    preset = MODEL_PRESETS[model]
    n_proc = preset["n_proc_per_node"]

    # Select scenarios
    if scenarios:
        fsdp_scenarios = {k: v for k, v in ALL_FSDP_SCENARIOS.items() if k in scenarios}
        if not fsdp_scenarios:
            log_message(LogLevel.ERROR, f"No valid scenarios in: {scenarios}")
            return
    else:
        fsdp_scenarios = ALL_FSDP_SCENARIOS

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_message(LogLevel.INFO, f"Model: {model} ({n_proc} GPUs)")
    log_message(LogLevel.INFO, f"Scenarios: {list(fsdp_scenarios.keys())}")

    # Seed checkpoint (uses torchtitan config, single process)
    seed_dir = out_path / "seed_checkpoint"
    seed_dir.mkdir(exist_ok=True)
    seed_config_path = str((SCRIPT_DIR / preset["torchtitan_config"]).resolve())
    _create_slurm_script(
        script_path=seed_dir / "seed.slurm",
        job_name=f"{model}_seed",
        config_path=seed_config_path,
        n_proc_per_node=n_proc,
        is_seed=True,
        model_flavor=preset["torchtitan_flavor"],
        qos=qos,
    )
    print(f"Created seed config at {seed_dir}")

    seed_checkpoint_path = str(seed_dir / "checkpoint" / "step-0")

    # Create scenario + backend directories
    backends = {
        "torchtitan": {
            "config": preset["torchtitan_config"],
            "use_hf_fsdp": False,
            "model_flavor": preset["torchtitan_flavor"],
        },
        "hf_fsdp": {
            "config": preset["hf_fsdp_config"],
            "use_hf_fsdp": True,
            "model_flavor": None,
        },
    }

    for scenario_name, scenario_opts in fsdp_scenarios.items():
        for backend_name, backend_opts in backends.items():
            job_dir = out_path / scenario_name / backend_name
            job_dir.mkdir(parents=True, exist_ok=True)

            config_path = str((SCRIPT_DIR / backend_opts["config"]).resolve())
            job_name = f"{model}_{scenario_name}_{backend_name}"

            _create_slurm_script(
                script_path=job_dir / "run.slurm",
                job_name=job_name,
                config_path=config_path,
                n_proc_per_node=n_proc,
                is_seed=False,
                seed_checkpoint_path=seed_checkpoint_path,
                use_hf_fsdp=backend_opts["use_hf_fsdp"],
                enable_cpu_offload=scenario_opts["enable_cpu_offload"],
                mixed_precision_param=scenario_opts["mixed_precision_param"],
                mixed_precision_reduce=scenario_opts["mixed_precision_reduce"],
                model_flavor=backend_opts["model_flavor"],
                qos=qos,
            )
            print(f"Created {job_name} at {job_dir}")

    log_message(LogLevel.SUCCESS, f"All configs created under {out_path}")


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

class Status(Enum):
    INIT = "init"
    PENDING = "pending"
    RUNNING = "running"
    FAIL = "fail"
    COMPLETED = "completed"


class Job:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path
        self.name = os.path.basename(root_path)

        # Find the slurm script
        slurm_files = [f for f in os.listdir(root_path) if f.endswith(".slurm")]
        if not slurm_files:
            raise FileNotFoundError(f"No .slurm file found in {root_path}")
        self.slurm_script = os.path.join(root_path, slurm_files[0])

        status_file = os.path.join(self.root_path, "status.txt")
        if not os.path.exists(status_file):
            with open(status_file, "w") as f:
                f.write(Status.INIT.value)
        self.status = self._read_status()

    def _read_status(self) -> Status:
        with open(os.path.join(self.root_path, "status.txt"), "r") as f:
            val = f.read().strip()
        return Status(val)

    def set_status(self, status: Status):
        with open(os.path.join(self.root_path, "status.txt"), "w") as f:
            f.write(status.value)
        self.status = status


def _find_job_dirs(inp_dir: str) -> list[str]:
    """Find all directories that contain a .slurm file."""
    dirs = []
    for root, _, files in os.walk(inp_dir):
        if any(f.endswith(".slurm") for f in files):
            dirs.append(os.path.abspath(root))
    return dirs


def submit_jobs(inp_dir: str, qos: str, only: str = None):
    job_dirs = _find_job_dirs(inp_dir)
    jobs = [Job(d) for d in job_dirs]

    if only:
        try:
            status_filter = Status(only)
            jobs = [j for j in jobs if j.status == status_filter]
        except ValueError:
            # Filter by name substring
            jobs = [j for j in jobs if only in j.root_path]

    # Filter out completed jobs
    jobs = [j for j in jobs if j.status != Status.COMPLETED]

    if not jobs:
        log_message(LogLevel.INFO, "No jobs to submit")
        return

    # Submit seed first, then the rest
    seed_jobs = [j for j in jobs if "seed" in j.name]
    other_jobs = [j for j in jobs if "seed" not in j.name]

    env_vars = os.environ.copy()

    for job in seed_jobs + other_jobs:
        log_message(LogLevel.INFO, f"Submitting {job.slurm_script}")
        result = subprocess.run(["sbatch", job.slurm_script], env=env_vars, capture_output=True, text=True)
        if result.returncode == 0:
            job.set_status(Status.PENDING)
            log_message(LogLevel.SUCCESS, f"Submitted: {result.stdout.strip()}", indent=1)
        else:
            log_message(LogLevel.ERROR, f"Failed: {result.stderr.strip()}", indent=1)


def check_status(inp_dir: str):
    job_dirs = _find_job_dirs(inp_dir)
    if not job_dirs:
        print(f"No jobs found in {inp_dir}")
        return

    status_counts = {s: 0 for s in Status}
    for d in job_dirs:
        job = Job(d)
        status_counts[job.status] += 1

    table = Table(title="Job Status Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")

    for status in Status:
        table.add_row(status.value.capitalize(), str(status_counts[status]))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(job_dirs)}[/bold]")

    console.print(table)

    # Also show per-job detail
    console.print()
    detail_table = Table(title="Job Details")
    detail_table.add_column("Job Path", style="cyan")
    detail_table.add_column("Status", justify="center")

    for d in sorted(job_dirs):
        job = Job(d)
        rel = os.path.relpath(d, inp_dir)
        status_style = {
            Status.INIT: "dim",
            Status.PENDING: "yellow",
            Status.RUNNING: "blue",
            Status.COMPLETED: "green",
            Status.FAIL: "bold red",
        }[job.status]
        detail_table.add_row(rel, f"[{status_style}]{job.status.value}[/{status_style}]")

    console.print(detail_table)


# ---------------------------------------------------------------------------
# Report: compare torchtitan vs hf_fsdp within each scenario
# ---------------------------------------------------------------------------

def report(inp_dir: str):
    import torch
    from dataclasses import dataclass, field
    from typing import List

    DEFAULT_LOSS_ATOL = 5e-2
    DEFAULT_LOSS_RTOL = 1e-5
    DEFAULT_GRAD_NORM_ATOL = 7e-1
    DEFAULT_GRAD_NORM_RTOL = 1e-5

    @dataclass
    class TrainingMetrics:
        steps: List[int] = field(default_factory=list)
        loss: List[float] = field(default_factory=list)
        grad_norm: List[float] = field(default_factory=list)

    def _extract_metrics(log_file: Path) -> TrainingMetrics:
        metrics = TrainingMetrics()
        with open(log_file, "r") as f:
            content = f.read()
        pattern = re.compile(
            r"step:\s*(\d+)\s*"
            r".*?loss:\s*(-?[0-9]+\.?[0-9]*)\s*"
            r".*?grad_norm:\s*([0-9]+\.?[0-9]*)\s*"
        )
        for match in pattern.finditer(content):
            loss = float(match.group(2))
            if loss == -1.0:
                continue
            metrics.steps.append(int(match.group(1)))
            metrics.loss.append(loss)
            metrics.grad_norm.append(float(match.group(3)))
        return metrics

    def _compare(baseline: TrainingMetrics, test: TrainingMetrics) -> tuple[bool, str]:
        if not baseline.loss or not test.loss:
            return False, "Unable to extract metrics"

        bl = torch.tensor(baseline.loss)
        tl = torch.tensor(test.loss)
        bg = torch.tensor(baseline.grad_norm)
        tg = torch.tensor(test.grad_norm)

        min_len = min(len(bl), len(tl))
        bl, tl = bl[:min_len], tl[:min_len]
        bg, tg = bg[:min_len], tg[:min_len]

        loss_pass = torch.allclose(bl, tl, atol=DEFAULT_LOSS_ATOL, rtol=DEFAULT_LOSS_RTOL)
        grad_pass = torch.allclose(bg, tg, atol=DEFAULT_GRAD_NORM_ATOL, rtol=DEFAULT_GRAD_NORM_RTOL)

        loss_max = torch.max(torch.abs(bl - tl)).item()
        grad_max = torch.max(torch.abs(bg - tg)).item()

        summary = f"Max loss diff: {loss_max:.2e}, Max grad_norm diff: {grad_max:.2e}"
        return (loss_pass and grad_pass), summary

    inp_path = Path(inp_dir)
    if not inp_path.exists():
        log_message(LogLevel.ERROR, f"Directory not found: {inp_path}")
        return

    console.print(Panel(
        "[bold cyan]FSDP Debug Report: torchtitan vs HF from_pretrained[/bold cyan]",
        expand=False, border_style="blue", padding=(1, 2),
    ))

    # Auto-discover scenario directories (any dir containing torchtitan/ and hf_fsdp/ subdirs)
    scenario_dirs = []
    for item in sorted(inp_path.iterdir()):
        if item.is_dir() and (item / "torchtitan").exists() and (item / "hf_fsdp").exists():
            scenario_dirs.append(item)

    if not scenario_dirs:
        log_message(LogLevel.ERROR, "No scenario directories found (expected dirs with torchtitan/ and hf_fsdp/ subdirs)")
        return

    results = []

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        tt_dir = scenario_dir / "torchtitan"
        hf_dir = scenario_dir / "hf_fsdp"

        # Find .out files
        tt_outs = list(tt_dir.glob("*.out"))
        hf_outs = list(hf_dir.glob("*.out"))

        if not tt_outs or not hf_outs:
            log_message(LogLevel.WARNING, f"{scenario_name}: Missing .out files (tt={len(tt_outs)}, hf={len(hf_outs)})")
            results.append((scenario_name, False, "Missing .out files"))
            continue

        tt_metrics = _extract_metrics(tt_outs[0])
        hf_metrics = _extract_metrics(hf_outs[0])

        passed, summary = _compare(tt_metrics, hf_metrics)

        if passed:
            log_message(LogLevel.TEST_PASS, f"{scenario_name} - {summary}")
        else:
            log_message(LogLevel.TEST_FAIL, f"{scenario_name} - {summary}")

        results.append((scenario_name, passed, summary))

        # Save diff
        diff_file = scenario_dir / "diff_torchtitan_vs_hf_fsdp.log"
        try:
            cmd = ["git", "diff", "--no-index", "--color=always", "--word-diff=color",
                   str(tt_outs[0]), str(hf_outs[0])]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode in [0, 1]:
                with open(diff_file, "w") as f:
                    f.write(result.stdout)
                log_message(LogLevel.INFO, f"Diff saved: {diff_file}", indent=1, dim=True)
        except Exception as e:
            log_message(LogLevel.WARNING, f"Diff failed: {e}", indent=1, dim=True)

    # Summary table
    console.print()
    summary_table = Table(title="[bold]Summary: torchtitan vs HF from_pretrained[/bold]",
                          show_header=True, header_style="bold magenta")
    summary_table.add_column("Scenario", style="cyan")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Details", style="dim")

    for name, passed, summary in results:
        status = "[bold green]✅ PASS[/bold green]" if passed else "[bold red]❌ FAIL[/bold red]"
        summary_table.add_row(name, status, summary)

    console.print(summary_table)

    total_passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    console.print()
    if total_passed == total and total > 0:
        log_message(LogLevel.SUCCESS, f"All {total} scenarios passed!")
    elif total > 0:
        log_message(LogLevel.WARNING, f"{total_passed}/{total} scenarios passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="FSDP debug runner for MoE non-tied model")
    subparsers = parser.add_subparsers(dest="action")

    create_parser = subparsers.add_parser("create_configs",
        help="Create output directory with SLURM scripts for FSDP scenarios")
    create_parser.add_argument("--out_dir", type=str, required=True,
        help="Output directory (e.g., ./outputs/moe_non_tied)")
    create_parser.add_argument("--model", type=str, default="debugmodel_moe",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset to use")
    create_parser.add_argument("--scenarios", type=str, nargs="+", default=None,
        choices=list(ALL_FSDP_SCENARIOS.keys()),
        help="FSDP scenarios to create (default: all)")
    create_parser.add_argument("--qos", type=str, default="normal",
        choices=["low", "normal", "high", "prod"])

    submit_parser = subparsers.add_parser("submit_jobs",
        help="Submit SLURM jobs")
    submit_parser.add_argument("--inp_dir", type=str, required=True)
    submit_parser.add_argument("--qos", type=str, default="normal",
        choices=["low", "normal", "high", "prod"])
    submit_parser.add_argument("--only", type=str, default=None,
        help="Filter: status value (init/fail) or name substring (e.g., 'fsdp_cpu_offload')")

    status_parser = subparsers.add_parser("check_status",
        help="Show job status summary")
    status_parser.add_argument("--inp_dir", type=str, required=True)

    report_parser = subparsers.add_parser("report",
        help="Compare torchtitan vs HF from_pretrained results")
    report_parser.add_argument("--inp_dir", type=str, required=True)

    args = parser.parse_args()

    if args.action == "create_configs":
        create_configs(args.out_dir, args.model, args.scenarios, args.qos)
    elif args.action == "submit_jobs":
        submit_jobs(args.inp_dir, args.qos, args.only)
    elif args.action == "check_status":
        check_status(args.inp_dir)
    elif args.action == "report":
        report(args.inp_dir)
    else:
        parser.print_help()
