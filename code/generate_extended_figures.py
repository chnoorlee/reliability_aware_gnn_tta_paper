"""Generate figures for the external-validity experiments.

Produces:
    figures/adversarial_ece.pdf   -- ECE under adversarial attacks (source vs full)
    figures/streaming_drift.pdf   -- accuracy across streaming steps per stream

Reads only stored result files under results/extended/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "extended"
FIG = ROOT / "paper" / "mypaper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = EXT / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["records"]


def adversarial_figure():
    recs = _load("adversarial_study")
    if not recs:
        return
    grouped = defaultdict(lambda: defaultdict(list))
    for r in recs:
        grouped[(r["attack"], r["intensity"])][r["method"]].append(r["ece"])
    labels, src, ent, full = [], [], [], []
    for (attack, intensity), methods in sorted(grouped.items()):
        labels.append(f"{attack}\n{intensity}")
        src.append(mean(methods["source_only"]))
        ent.append(mean(methods["entropy_all_nodes"]))
        full.append(mean(methods["full_method"]))
    x = range(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([i - w for i in x], src, width=w, label="Source only", color="#d62728")
    ax.bar(list(x), ent, width=w, label="Entropy all nodes", color="#ff7f0e")
    ax.bar([i + w for i in x], full, width=w, label="Full method", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Calibration under adversarial attack")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "adversarial_ece.pdf")
    plt.close(fig)
    print("wrote adversarial_ece.pdf")


def streaming_figure():
    recs = _load("streaming_tta_study")
    if not recs:
        return
    streams = sorted({r["stream"] for r in recs})
    fig, axes = plt.subplots(1, len(streams), figsize=(11, 3.2), sharey=True)
    if len(streams) == 1:
        axes = [axes]
    for ax, stream in zip(axes, streams):
        sub = [r for r in recs if r["stream"] == stream]
        by_method = defaultdict(lambda: defaultdict(list))
        for r in sub:
            by_method[r["method"]][r["step"]].append(r["accuracy"])
        for method, color in [
            ("source_only", "#1f77b4"),
            ("entropy_all_nodes", "#ff7f0e"),
            ("full_method", "#2ca02c"),
        ]:
            steps = sorted(by_method[method])
            ys = [mean(by_method[method][s]) for s in steps]
            ax.plot(steps, ys, marker="o", label=method, color=color)
        ax.set_title(stream.replace("_", " "), fontsize=9)
        ax.set_xlabel("stream step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Streaming / continual TTA: accuracy across accumulating shifts")
    fig.tight_layout()
    fig.savefig(FIG / "streaming_drift.pdf")
    plt.close(fig)
    print("wrote streaming_drift.pdf")


if __name__ == "__main__":
    adversarial_figure()
    streaming_figure()
