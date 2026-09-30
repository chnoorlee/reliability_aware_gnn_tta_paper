"""Post-hoc calibration baselines requested by reviewers.

The main method is label-free at test time.  This script adds a conventional
source-validation temperature-scaling baseline for context: a single scalar
temperature is fit on the source validation split and then applied unchanged to
each shifted target graph.

Usage:
    python calibration_baselines.py
    python calibration_baselines.py --render
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F

from _np_bridge import evaluate
from exp_common import shift_bundle, train_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_torch" / "supplementary"
JSON_PATH = OUT / "temperature_scaling.json"
CSV_PATH = OUT / "temperature_scaling.csv"

CONDITIONS = [
    ("clean", 0.0),
    ("feature_noise", 0.45),
    ("edge_drop", 0.35),
    ("edge_add", 0.35),
    ("homophily_shift", 0.25),
    ("homophily_shift", 0.50),
]


def fit_temperature(model, base, max_iter=80):
    """Fit one positive scalar temperature on source validation labels."""
    model.eval()
    with torch.no_grad():
        logits = model(base.x, base.edge_index).detach()
    val_logits = logits[base.val_mask]
    val_labels = base.y[base.val_mask]
    log_t = torch.zeros((), device=val_logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.cross_entropy(val_logits / temp, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float(torch.exp(log_t).clamp(0.05, 20.0).detach().cpu())
    before = float(F.cross_entropy(val_logits, val_labels).detach().cpu())
    after = float(F.cross_entropy(val_logits / temp, val_labels).detach().cpu())
    return temp, before, after


@torch.no_grad()
def eval_logits(model, sb, temperature=1.0):
    model.eval()
    logits = model(sb.x, sb.edge_index) / float(temperature)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)


def run(seeds=(0, 1, 2, 3, 4)):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=360, use_bn=False)
        temperature, val_nll_before, val_nll_after = fit_temperature(model, base)
        for shift, intensity in CONDITIONS:
            sb = shift_bundle(base, seed, shift, intensity)
            for method, temp in [
                ("source_only", 1.0),
                ("temperature_scaling_source_val", temperature),
            ]:
                metrics = eval_logits(model, sb, temperature=temp)
                records.append({
                    "seed": seed,
                    "shift": shift,
                    "intensity": intensity,
                    "method": method,
                    "temperature": temp,
                    "val_nll_before": val_nll_before,
                    "val_nll_after": val_nll_after,
                    "accuracy": float(metrics["accuracy"]),
                    "ece": float(metrics["ece"]),
                    "nll": float(metrics["nll"]),
                    "brier": float(metrics["brier"]),
                })
            print(f"[temperature_scaling] seed={seed} {shift}/{intensity} T={temperature:.3f}")
    JSON_PATH.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        writer.writeheader()
        writer.writerows(records)
    render(records)


def _fmt(vals, digits=4):
    vals = list(vals)
    if len(vals) > 1:
        return f"{mean(vals):.{digits}f}\\pm{pstdev(vals):.{digits}f}"
    return f"{mean(vals):.{digits}f}" if vals else "n/a"


def _load_records():
    if not JSON_PATH.exists():
        raise SystemExit(f"missing {JSON_PATH}; run without --render first")
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))["records"]


def render(records=None):
    records = records or _load_records()
    groups = defaultdict(list)
    for r in records:
        groups[(r["shift"], float(r["intensity"]), r["method"])].append(r)

    temps = [r["temperature"] for r in records if r["method"] == "temperature_scaling_source_val"]
    val_before = [r["val_nll_before"] for r in records if r["method"] == "temperature_scaling_source_val"]
    val_after = [r["val_nll_after"] for r in records if r["method"] == "temperature_scaling_source_val"]
    print("Temperature scaling summary")
    print(f"T = {_fmt(temps, 3)}; source-val NLL {_fmt(val_before)} -> {_fmt(val_after)}")
    print("Shift & Method & Accuracy & ECE & NLL & Brier \\\\")
    for shift, intensity in CONDITIONS:
        for method in ["source_only", "temperature_scaling_source_val"]:
            sub = groups[(shift, float(intensity), method)]
            print(
                f"{shift} ({intensity:.2f}) & {method} & "
                f"${_fmt([r['accuracy'] for r in sub])}$ & "
                f"${_fmt([r['ece'] for r in sub])}$ & "
                f"${_fmt([r['nll'] for r in sub])}$ & "
                f"${_fmt([r['brier'] for r in sub])}$ \\\\"
            )
    print("% ECE deltas: temperature scaling minus source-only")
    for shift, intensity in CONDITIONS:
        src = groups[(shift, float(intensity), "source_only")]
        tmp = groups[(shift, float(intensity), "temperature_scaling_source_val")]
        print(
            f"% {shift}/{intensity:.2f}: "
            f"dECE={mean(r['ece'] for r in tmp) - mean(r['ece'] for r in src):+.4f}, "
            f"dNLL={mean(r['nll'] for r in tmp) - mean(r['nll'] for r in src):+.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render()
    else:
        run()


if __name__ == "__main__":
    main()
