"""Aggregate the extended experiment outputs (WebKB, adversarial, streaming, large-scale)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


EXT = Path(__file__).resolve().parents[1] / "results" / "extended"


def _load(name):
    p = EXT / f"{name}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)["records"]


def _ms(vals):
    vals = list(vals)
    if len(vals) > 1:
        return f"{mean(vals):.4f}±{stdev(vals):.4f}"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return "n/a"


def aggregate_webkb():
    records = _load("real_webkb_study")
    if not records:
        print("\n[real_webkb_study] no records")
        return
    grouped = defaultdict(lambda: defaultdict(list))
    homophily = {}
    for r in records:
        grouped[(r["dataset"], r["shift"], r["intensity"])][r["method"]].append(r)
        homophily[r["dataset"]] = r.get("homophily", 0.0)
    print("\n=== Real WebKB (Texas/Cornell/Wisconsin) — mean±std over 5 seeds ===")
    print(f"{'dataset':<10} {'hG':<7} {'shift':<18} {'method':<18} {'acc':<16} {'ece':<16}")
    for (dataset, shift, intensity), methods in sorted(grouped.items()):
        for method, rows in methods.items():
            print(
                f"{dataset:<10} {homophily[dataset]:<7.3f} {shift+'/'+str(intensity):<18} "
                f"{method:<18} {_ms(r['accuracy'] for r in rows):<16} {_ms(r['ece'] for r in rows):<16}"
            )


def aggregate_adversarial():
    records = _load("adversarial_study")
    if not records:
        print("\n[adversarial_study] no records")
        return
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r["attack"], r["intensity"])][r["method"]].append(r)
    print("\n=== Adversarial shift — mean±std over 5 seeds ===")
    print(f"{'attack':<16} {'intensity':<10} {'method':<18} {'acc':<16} {'ece':<16}")
    for (attack, intensity), methods in sorted(grouped.items()):
        for method, rows in methods.items():
            print(
                f"{attack:<16} {intensity:<10} {method:<18} "
                f"{_ms(r['accuracy'] for r in rows):<16} {_ms(r['ece'] for r in rows):<16}"
            )


def aggregate_streaming():
    records = _load("streaming_tta_study")
    if not records:
        print("\n[streaming_tta_study] no records")
        return
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r["stream"], r["step"], r["intensity"])][r["method"]].append(r)
    print("\n=== Streaming / continual TTA — mean acc over 5 seeds, per step ===")
    print(f"{'stream':<22} {'step':<5} {'intensity':<10} {'method':<18} {'acc':<16} {'trig%':<6}")
    for (stream, step, intensity), methods in sorted(grouped.items()):
        for method, rows in methods.items():
            trig = mean(1.0 if r.get("detector_triggered") else 0.0 for r in rows) * 100 if method == "full_method" else 0.0
            print(
                f"{stream:<22} {step:<5} {intensity:<10} {method:<18} "
                f"{_ms(r['accuracy'] for r in rows):<16} {trig:<6.0f}"
            )


def aggregate_large_scale():
    records = _load("large_scale_study")
    if not records:
        print("\n[large_scale_study] no records yet")
        return
    grouped = defaultdict(lambda: defaultdict(list))
    nodes = {}
    for r in records:
        grouped[(r["dataset"], r["shift"], r["intensity"])][r["method"]].append(r)
        nodes[r["dataset"]] = r.get("num_nodes", 0)
    print("\n=== Large-scale real graphs (Amazon/Coauthor subgraphs) — mean±std ===")
    print(f"{'dataset':<18} {'n':<6} {'shift':<18} {'method':<18} {'acc':<16} {'ece':<16}")
    for (dataset, shift, intensity), methods in sorted(grouped.items()):
        for method, rows in methods.items():
            print(
                f"{dataset:<18} {nodes[dataset]:<6} {shift+'/'+str(intensity):<18} "
                f"{method:<18} {_ms(r['accuracy'] for r in rows):<16} {_ms(r['ece'] for r in rows):<16}"
            )


def main():
    aggregate_webkb()
    aggregate_adversarial()
    aggregate_streaming()
    aggregate_large_scale()


if __name__ == "__main__":
    main()
