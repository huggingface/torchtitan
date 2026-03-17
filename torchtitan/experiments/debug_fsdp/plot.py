"""
Plot training metrics from FSDP debug runs.

Usage:
    python plot.py --inp_dir ./outputs/llama3_8b
    python plot.py --inp_dir ./outputs/llama3_8b --scenarios fsdp fsdp_mixed_precision
"""

import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@dataclass
class Metrics:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    memory_gib: list[float] = field(default_factory=list)
    tps: list[float] = field(default_factory=list)
    tflops: list[float] = field(default_factory=list)
    mfu: list[float] = field(default_factory=list)


def extract_metrics(log_file: Path) -> Metrics:
    metrics = Metrics()
    content = log_file.read_text()

    # Strip ANSI color codes
    content = re.sub(r"\x1b\[[0-9;]*m", "", content)

    pattern = re.compile(
        r"step:\s*(\d+)\s*"
        r".*?loss:\s*(-?[0-9]+\.?[0-9]*)\s*"
        r".*?grad_norm:\s*([0-9]+\.?[0-9]*)\s*"
        r".*?memory:\s*([0-9]+\.?[0-9]*)GiB"
        r".*?tps:\s*([0-9,]+)\s*"
        r".*?tflops:\s*([0-9]+\.?[0-9]*)\s*"
        r".*?mfu:\s*([0-9]+\.?[0-9]*)%"
    )

    for match in pattern.finditer(content):
        loss = float(match.group(2))
        if loss == -1.0:
            continue
        metrics.steps.append(int(match.group(1)))
        metrics.loss.append(loss)
        metrics.grad_norm.append(float(match.group(3)))
        metrics.memory_gib.append(float(match.group(4)))
        metrics.tps.append(float(match.group(5).replace(",", "")))
        metrics.tflops.append(float(match.group(6)))
        metrics.mfu.append(float(match.group(7)))

    return metrics


def find_out_file(directory: Path) -> Path | None:
    outs = list(directory.glob("*.out"))
    return outs[0] if outs else None


def plot_comparison(inp_dir: Path, scenarios: list[str] | None, out_file: str):
    # Discover scenarios
    all_scenarios = []
    for d in sorted(inp_dir.iterdir()):
        if d.is_dir() and (d / "torchtitan").exists() and (d / "hf_fsdp").exists():
            all_scenarios.append(d.name)

    if scenarios:
        all_scenarios = [s for s in all_scenarios if s in scenarios]

    if not all_scenarios:
        print("No scenarios found")
        return

    # Collect metrics
    data = {}  # {scenario: {backend: Metrics}}
    for scenario in all_scenarios:
        data[scenario] = {}
        for backend in ("torchtitan", "hf_fsdp"):
            out_file_path = find_out_file(inp_dir / scenario / backend)
            if out_file_path:
                m = extract_metrics(out_file_path)
                if m.steps:
                    data[scenario][backend] = m

    # Filter to scenarios with at least one backend
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("No metrics found in any scenario")
        return

    n_scenarios = len(data)
    metric_names = [
        ("loss", "Loss"),
        ("grad_norm", "Grad Norm"),
        ("memory_gib", "Memory (GiB)"),
        ("tps", "Tokens/sec"),
    ]
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(
        n_metrics, n_scenarios,
        figsize=(6 * n_scenarios, 4 * n_metrics),
        squeeze=False,
    )

    colors = {"torchtitan": "#1f77b4", "hf_fsdp": "#ff7f0e"}
    labels = {"torchtitan": "TorchTitan", "hf_fsdp": "HF from_pretrained"}

    for col, scenario in enumerate(data):
        for row, (attr, title) in enumerate(metric_names):
            ax = axes[row][col]

            for backend, metrics in data[scenario].items():
                values = getattr(metrics, attr)
                if values:
                    ax.plot(
                        metrics.steps, values,
                        color=colors[backend],
                        label=labels[backend],
                        linewidth=1.0,
                        alpha=0.85,
                    )

            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            if row == 0:
                ax.set_title(scenario.replace("_", " ").upper(), fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_path = inp_dir / out_file
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FSDP debug training metrics")
    parser.add_argument("--inp_dir", type=str, required=True)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Filter to specific scenarios")
    parser.add_argument("--out_file", type=str, default="metrics.svg",
                        help="Output filename (saved in inp_dir)")
    args = parser.parse_args()

    plot_comparison(Path(args.inp_dir), args.scenarios, args.out_file)
