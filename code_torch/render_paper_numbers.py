"""Render every paper table body from the PyTorch-Geometric result tree.

Replaces the old "regenerate all of results.tex" approach (results.tex is now a
hand-crafted document): this script reads the stored ``results_torch/`` files
and prints, per table, the exact LaTeX rows in the current table formats so the
table bodies in ``sections/results.tex`` / ``sections/appendix.tex`` can be
swapped value-for-value.  Every number is computed from a stored result file --
nothing is invented.  Missing inputs are skipped with a notice so the script
can run incrementally while experiments are still finishing.

Usage:  python render_paper_numbers.py [table ...]
        (no args = render everything available)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "results_torch"

SHIFT_LABEL = {"clean": "Clean", "feature_noise": "Feature noise", "edge_drop": "Edge drop",
               "edge_add": "Edge add", "degree_shift": "Degree shift", "homophily_shift": "Homophily shift"}
COND_ORDER = {"clean": 0, "feature_noise": 1, "edge_drop": 2, "edge_add": 3, "degree_shift": 4, "homophily_shift": 5}
IN_ENVELOPE = [("clean", 0.0), ("feature_noise", 0.20), ("feature_noise", 0.45), ("edge_drop", 0.15),
               ("edge_drop", 0.35), ("edge_add", 0.15), ("edge_add", 0.35), ("degree_shift", 0.35)]


def _cond_label(shift, intensity):
    base = SHIFT_LABEL.get(shift, shift)
    return base if shift == "clean" else f"{base} ({float(intensity):.2f})"


def _csv_rows(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json_records(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _ms(vals, digits=4, bold=False):
    """math-mode mean±std body (no surrounding $); bold wraps the mean in \\mathbf."""
    vals = list(vals)
    if not vals:
        return "n/a"
    m = f"{mean(vals):.{digits}f}"
    if bold:
        m = f"\\mathbf{{{m}}}"
    if len(vals) > 1:
        return f"{m}\\pm{stdev(vals):.{digits}f}"
    return m


def _mpm(m, s, digits=4, bold=False):
    """text-mode mean$\\pm$std (tab:main style); bold uses \\textbf on the mean."""
    mtxt = f"{m:.{digits}f}"
    if bold:
        mtxt = f"\\textbf{{{mtxt}}}"
    return f"{mtxt}$\\pm${s:.{digits}f}"


def _bold(text, do):
    return f"\\textbf{{{text}}}" if do else text


def section(title):
    print("\n" + "=" * 100)
    print(f"### {title}")
    print("=" * 100)


# ------------------------------------------------------------------ tab:main
def tab_main():
    rows = _csv_rows(RT / "summary.csv")
    det = _json_records(RT / "detector_main.json")
    if rows is None:
        print("[tab:main] missing results_torch/summary.csv"); return
    rows = [r for r in rows if r["dataset"] == "synthetic"]
    by = {(r["shift"], float(r["intensity"]), r["method"]): r for r in rows}
    conds = sorted({(r["shift"], float(r["intensity"])) for r in rows},
                   key=lambda c: (COND_ORDER.get(c[0], 9), c[1]))
    # method sets per condition: representative conditions also show Tent/EATA/Matcha
    rep = {("clean", 0.0), ("edge_add", 0.35), ("homophily_shift", 0.25), ("homophily_shift", 0.50)}
    label = {"source_only": "Source only", "tent_entropy": "Tent", "eata_filter": "EATA",
             "matcha_reliable": "Matcha", "entropy_all_nodes": "Entropy all nodes",
             "full_method": "Full method"}
    section("tab:main  (operating envelope; bold best per column within block)")
    for shift, inten in conds:
        methods = (["source_only", "tent_entropy", "eata_filter", "matcha_reliable",
                    "entropy_all_nodes", "full_method"] if (shift, inten) in rep
                   else ["source_only", "entropy_all_nodes", "full_method"])
        block = []
        for m in methods:
            r = by.get((shift, inten, m))
            if r is None:
                continue
            block.append((label[m], float(r["accuracy_mean"]), float(r["accuracy_std"]),
                          float(r["macro_f1_mean"]), float(r["macro_f1_std"]),
                          float(r["ece_mean"]), float(r["ece_std"]),
                          float(r["nll_mean"]), float(r["nll_std"])))
        if shift == "homophily_shift" and abs(inten - 0.50) < 1e-9:
            # rename plain full method; add detector row from the dedicated run
            block = [(b[0] if b[0] != "Full method" else "Full method (no detector)",) + tuple(b[1:])
                     for b in block]
            if det:
                for mname, lab in [("full_method_detector_fixed", "Full method + detector (fixed $\\Delta^\\star{=}0.05$)"),
                                   ("full_method_detector_auto", "Full method + detector (auto $\\Delta^\\star$)")]:
                    d = [r for r in det if r["shift"] == "homophily_shift"
                         and abs(r["intensity"] - 0.50) < 1e-9 and r.get("method") == mname]
                    if d:
                        block.append((lab,
                                      mean(r["accuracy"] for r in d), stdev(r["accuracy"] for r in d),
                                      mean(r["macro_f1"] for r in d), stdev(r["macro_f1"] for r in d),
                                      mean(r["ece"] for r in d), stdev(r["ece"] for r in d),
                                      mean(r["nll"] for r in d), stdev(r["nll"] for r in d)))
                        trig = sum(r["triggered"] for r in d)
                        steps = [r["trigger_step"] for r in d if r["trigger_step"] is not None]
                        print(f"% {mname}: trigger {trig}/{len(d)} seeds, steps={steps}")
        best_acc = max(b[1] for b in block); best_f1 = max(b[3] for b in block)
        best_ece = min(b[5] for b in block); best_nll = min(b[7] for b in block)
        for name, am, as_, fm, fs, em, es, nm, ns in block:
            cells = [
                _mpm(am, as_, bold=abs(am - best_acc) < 5e-5),
                _mpm(fm, fs, bold=abs(fm - best_f1) < 5e-5),
                _mpm(em, es, bold=abs(em - best_ece) < 5e-5),
                _mpm(nm, ns, bold=abs(nm - best_nll) < 5e-5),
            ]
            print(f"{_cond_label(shift, inten)} & {name} & " + " & ".join(cells) + " \\\\")
        print("\\cmidrule(lr){1-6}")


# ------------------------------------------------------------------ prose numbers
def prose_main():
    rows = _csv_rows(RT / "summary.csv")
    if rows is None:
        return
    rows = [r for r in rows if r["dataset"] == "synthetic"]
    by = {(r["shift"], float(r["intensity"]), r["method"]): r for r in rows}
    section("prose numbers (sec:res-calibration / runtime / ablation text)")
    gains = []
    for shift, inten in IN_ENVELOPE + [("homophily_shift", 0.25)]:
        s = by.get((shift, inten, "source_only")); f = by.get((shift, inten, "full_method"))
        if s and f and float(s["ece_mean"]) > 0:
            rel = (float(s["ece_mean"]) - float(f["ece_mean"])) / float(s["ece_mean"])
            gains.append((f"{shift}/{inten}", float(s["ece_mean"]), float(f["ece_mean"]), rel))
    for g in gains:
        print(f"  ECE {g[0]}: source {g[1]:.4f} -> full {g[2]:.4f}  ({100*g[3]:+.1f}% reduction)")
    in_env_gains = [g[3] for g in gains if not g[0].startswith("homophily")]
    print(f"  in-envelope relative ECE reduction range: {100*min(in_env_gains):.1f}% .. {100*max(in_env_gains):.1f}%")
    fm = [r for r in rows if r["method"] == "full_method" and (r["shift"], float(r["intensity"])) in IN_ENVELOPE]
    print(f"  full_method mean reliability={mean(float(r['mean_reliability_mean']) for r in fm):.2f} "
          f"selected fraction={mean(float(r['selected_fraction_mean']) for r in fm):.2f}")
    ent = [r for r in rows if r["method"] == "entropy_all_nodes" and (r["shift"], float(r["intensity"])) in IN_ENVELOPE]
    rt_full = mean(float(r["runtime_seconds_mean"]) for r in fm)
    rt_ent = mean(float(r["runtime_seconds_mean"]) for r in ent)
    print(f"  in-envelope runtime: full={rt_full:.4f}s entropy={rt_ent:.4f}s ratio={rt_full/rt_ent:.1f}x")
    nost = [r for r in rows if r["method"] == "no_structural_stability" and (r["shift"], float(r["intensity"])) in IN_ENVELOPE]
    if nost:
        print(f"  no_structural_stability runtime={mean(float(r['runtime_seconds_mean']) for r in nost):.4f}s")
    # out-of-envelope numbers
    for m in ["source_only", "entropy_all_nodes", "full_method"]:
        r = by.get(("homophily_shift", 0.50, m))
        if r:
            print(f"  homophily0.50 {m}: acc={float(r['accuracy_mean']):.4f} ece={float(r['ece_mean']):.4f}")


# ------------------------------------------------------------------ tab:ablation
def tab_ablation():
    rows = _csv_rows(RT / "summary.csv")
    if rows is None:
        return
    rows = [r for r in rows if r["dataset"] == "synthetic" and (r["shift"], float(r["intensity"])) in IN_ENVELOPE]
    label = {"full_method": "Full method", "no_neighborhood_agreement": "No neighborhood agreement",
             "no_structural_stability": "No structural stability", "no_calibration_loss": "No calibration loss",
             "no_anti_forgetting": "No anti-forgetting"}
    section("tab:ablation  (8 in-envelope conditions averaged)")
    for m, lab in label.items():
        sub = [r for r in rows if r["method"] == m]
        if not sub:
            continue
        print(f"{lab} & {mean(float(r['accuracy_mean']) for r in sub):.4f} & "
              f"{mean(float(r['ece_mean']) for r in sub):.4f} & "
              f"{mean(float(r['brier_mean']) for r in sub):.4f} & "
              f"{mean(float(r['runtime_seconds_mean']) for r in sub):.4f} \\\\")


# ------------------------------------------------------------------ tab:public
def tab_public():
    rows = _csv_rows(RT / "public_benchmark" / "summary.csv")
    if rows is None:
        print("[tab:public] missing"); return
    label = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes", "full_method": "Full method"}
    section("tab:public-benchmark")
    for ds in ["cora", "citeseer", "pubmed"]:
        for m, lab in label.items():
            sub = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            clean = [r for r in sub if r["shift"] == "clean"]
            shifted = [r for r in sub if r["shift"] != "clean"]
            if not clean or not shifted:
                continue
            print(f"{ds} & {lab} & {mean(float(r['accuracy_mean']) for r in clean):.4f} & "
                  f"{mean(float(r['accuracy_mean']) for r in shifted):.4f} & "
                  f"{mean(float(r['ece_mean']) for r in shifted):.4f} & "
                  f"{mean(float(r['runtime_seconds_mean']) for r in shifted):.4f} \\\\")


# ------------------------------------------------------------------ tab:fair
def tab_fair():
    recs = _json_records(RT / "fair_comparison" / "fair_comparison.json")
    if recs is None:
        print("[tab:fair] missing"); return
    label = {"source_only": "Source only", "tent": "Tent (full, BN)", "eata": "EATA (full, BN+Fisher)",
             "matcha": "Matcha (full, mask)", "gtrans": "GTrans (full, transform)",
             "full_method": "Full method (ours)"}
    section("tab:fair  (in-env rows average per-dataset STRUCTURAL conditions, i.e. clean+edge; out-env = synthetic homophily 0.50)")
    groups = []
    for ds in ["cora", "citeseer", "pubmed", "coauthor_cs", "amazon_photo"]:
        groups.append((f"{ds} (in-env)",
                       [r for r in recs if r["dataset"] == ds and not r["shift"].startswith("feature")]))
    groups.append(("Synthetic (out-env h=0.5)",
                   [r for r in recs if r["dataset"] == "synthetic" and abs(float(r["intensity"]) - 0.50) < 1e-9]))
    for gname, sub in groups:
        if not sub:
            continue
        block = []
        for m in label:
            mrows = [r for r in sub if r["method"] == m]
            if not mrows:
                continue
            block.append((label[m], mean(r["accuracy"] for r in mrows), mean(r["ece"] for r in mrows),
                          mean(r["brier"] for r in mrows), mean(r["runtime_seconds"] for r in mrows)))
        best_acc = max(b[1] for b in block); best_ece = min(b[2] for b in block); best_br = min(b[3] for b in block)
        for name, a, e, b, rt in block:
            print(f"{gname} & {name} & "
                  f"{_bold(f'{a:.4f}', abs(a-best_acc)<5e-5)} & {_bold(f'{e:.4f}', abs(e-best_ece)<5e-5)} & "
                  f"{_bold(f'{b:.4f}', abs(b-best_br)<5e-5)} & {rt:.2f} \\\\")
        print("\\cmidrule(lr){1-6}")


# ------------------------------------------------------------------ tab:significance
def tab_significance():
    rows = _csv_rows(RT / "significance.csv")
    if rows is None:
        print("[tab:significance] missing"); return
    section("tab:significance  (synthetic, full vs source/entropy)")
    keep_b = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes"}
    picked = [r for r in rows if r["method_a"] == "full_method" and r["method_b"] in keep_b]
    picked = sorted(picked, key=lambda r: (r["shift"], float(r["intensity"]), r["method_b"]))
    for r in picked:
        p = float(r["accuracy_p"] or 1.0)
        marker = "$^*$" if p < 0.05 else ""
        print(f"synthetic & {_cond_label(r['shift'], r['intensity'])} & Full vs. {keep_b[r['method_b']]} & "
              f"{float(r['accuracy_diff']):.4f}{marker} & {p:.4f} & {float(r['ece_diff']):.4f} \\\\")


# ------------------------------------------------------------------ tab:assumption
def tab_assumption():
    recs = _json_records(RT / "supplementary" / "assumption_verification.json")
    if recs is None:
        print("[tab:assumption] missing"); return
    section("tab:assumption")
    by = defaultdict(list)
    for r in recs:
        by[(r["shift"], float(r["intensity"]))].append(r)
    for (shift, inten) in sorted(by, key=lambda c: (COND_ORDER.get(c[0], 9), c[1])):
        rs = by[(shift, inten)]
        cov = [r["cov_r_e"] for r in rs]
        gap = mean(r["error_rate_bottom50_reliability"] - r["error_rate_top50_reliability"] for r in rs)
        hg = mean(r["estimated_homophily"] for r in rs)
        holds = all(r["assumption_holds"] for r in rs)
        print(f"{SHIFT_LABEL[shift]} & {inten:.2f} & ${mean(cov):.4f}\\pm {stdev(cov):.4f}$ & "
              f"${gap:+.3f}$ & ${hg:.3f}$ & {'yes' if holds else 'no'} \\\\")


# ------------------------------------------------------------------ tab:backbone
def tab_backbone():
    recs = _json_records(RT / "supplementary" / "backbone_study.json")
    if recs is None:
        print("[tab:backbone-study] missing"); return
    section("tab:backbone-study")
    conds = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
             ("edge_add", 0.35), ("homophily_shift", 0.25), ("homophily_shift", 0.50)]
    for bb in ["gcn", "gat", "graphsage", "appnp"]:
        for shift, inten in conds:
            cells = {}
            for m in ["source_only", "entropy_all_nodes", "full_method"]:
                rs = [r for r in recs if r["backbone"] == bb and r["shift"] == shift
                      and abs(float(r["intensity"]) - inten) < 1e-9 and r["method"] == m]
                if rs:
                    cells[m] = (mean(r["accuracy"] for r in rs), mean(r["ece"] for r in rs))
            if len(cells) < 3:
                continue
            best_acc = max(v[0] for v in cells.values()); best_ece = min(v[1] for v in cells.values())
            parts = []
            for m in ["source_only", "entropy_all_nodes", "full_method"]:
                a, e = cells[m]
                am = f"\\mathbf{{{a:.4f}}}" if abs(a - best_acc) < 5e-5 else f"{a:.4f}"
                em = f"\\mathbf{{{e:.4f}}}" if abs(e - best_ece) < 5e-5 else f"{e:.4f}"
                parts.append(f"${am}$")
                parts.append(f"${em}$")
            name = {"gcn": "GCN", "gat": "GAT", "graphsage": "GraphSAGE", "appnp": "APPNP"}[bb]
            print(f"{name} & {_cond_label(shift, inten)} & " + " & ".join(parts) + " \\\\")
        print("\\midrule")


# ------------------------------------------------------------------ tab:large-scale
def tab_large():
    recs = _json_records(RT / "extended" / "large_scale_study.json")
    if recs is None:
        print("[tab:large-scale] missing"); return
    section("tab:large-scale  (structural conditions; feature_noise reported separately)")
    label = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes", "full_method": "Full method"}
    name = {"coauthor_cs": "Coauthor CS", "amazon_photo": "Amazon Photo", "amazon_computers": "Amazon Computers"}
    for ds in ["coauthor_cs", "amazon_photo", "amazon_computers"]:
        for shift, inten in [("clean", 0.0), ("edge_add", 0.10), ("edge_drop", 0.15), ("feature_noise", 0.10)]:
            block = []
            for m in label:
                rs = [r for r in recs if r["dataset"] == ds and r["shift"] == shift
                      and abs(float(r["intensity"]) - inten) < 1e-9 and r["method"] == m]
                if rs:
                    block.append((label[m], [r["accuracy"] for r in rs], [r["ece"] for r in rs]))
            if not block:
                continue
            best_acc = max(mean(b[1]) for b in block); best_ece = min(mean(b[2]) for b in block)
            for lab, accs, eces in block:
                print(f"{name[ds]} & {_cond_label(shift, inten)} & {lab} & "
                      f"${_ms(accs, bold=abs(mean(accs)-best_acc)<5e-5)}$ & "
                      f"${_ms(eces, bold=abs(mean(eces)-best_ece)<5e-5)}$ \\\\")
            print("\\cmidrule(lr){1-5}")


# ------------------------------------------------------------------ tab:webkb
def tab_webkb():
    recs = _json_records(RT / "extended" / "real_webkb_study.json")
    if recs is None:
        print("[tab:webkb] missing"); return
    section("tab:webkb (all conditions x methods; select rows as in current table)")
    label = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes", "full_method": "Full method"}
    hs = {}
    for r in recs:
        hs[r["dataset"]] = r["homophily"]
    for ds in ["texas", "cornell", "wisconsin"]:
        for shift, inten in [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.10), ("edge_add", 0.05)]:
            for m, lab in label.items():
                rs = [r for r in recs if r["dataset"] == ds and r["shift"] == shift
                      and abs(float(r["intensity"]) - inten) < 1e-9 and r["method"] == m]
                if rs:
                    print(f"{ds.capitalize()} & {hs[ds]:.3f} & {_cond_label(shift, inten)} & {lab} & "
                          f"${_ms([r['accuracy'] for r in rs])}$ & ${_ms([r['ece'] for r in rs])}$ \\\\")


# ------------------------------------------------------------------ tab:webkb-assumption
def tab_webkb_assumption():
    recs = _json_records(RT / "extended" / "webkb_assumption.json")
    if recs is None:
        print("[tab:webkb-assumption] missing"); return
    section("tab:webkb-assumption")
    by = defaultdict(list)
    for r in recs:
        by[r["dataset"]].append(r)
    for ds in ["texas", "cornell", "wisconsin"]:
        rs = by[ds]
        n_neg = sum(1 for r in rs if r["assumption_holds"])
        print(f"{ds.capitalize()} & ${mean(r['cov_r_e'] for r in rs):.4f}$ & {n_neg}/{len(rs)} & "
              f"${mean(r['corr_confidence_err'] for r in rs):+.3f}$ & "
              f"${mean(r['corr_agreement_err'] for r in rs):+.3f}$ & "
              f"${mean(r['corr_stability_err'] for r in rs):+.3f}$ & "
              f"${mean(r['true_homophily'] for r in rs):.3f}$ & "
              f"${mean(r['pseudo_homophily'] for r in rs):.3f}$ \\\\")
    print(f"% feature-homophily mean: " +
          ", ".join(f"{ds}={mean(r['feature_homophily'] for r in by[ds]):.3f}" for ds in by))


# ------------------------------------------------------------------ tab:boundary
def tab_boundary():
    recs = _json_records(RT / "extended" / "boundary_homophily.json")
    if recs is None:
        print("[tab:boundary] missing"); return
    section("tab:boundary-homophily")
    label = {"source_only": "Source only", "tent_entropy": "Tent", "eata_filter": "EATA",
             "matcha_reliable": "Matcha", "full_method": "Full method"}
    hg = {"actor": 0.38, "film": 0.45}
    by = defaultdict(list)
    for r in recs:
        by[(r["dataset"], r["method"])].append(r)
    for ds in ["actor", "film"]:
        for m, lab in label.items():
            rs = by.get((ds, m), [])
            if rs:
                print(f"{ds.capitalize()} ({hg[ds]:.2f}) & {lab} & "
                      f"{mean(r['clean_acc'] for r in rs):.4f} & "
                      f"{mean(r['shifted_acc'] for r in rs):.4f} & "
                      f"{mean(r['ece'] for r in rs):.4f} \\\\")
        print("\\cmidrule(lr){1-5}")


# ------------------------------------------------------------------ tab:streaming + dual
def tab_streaming():
    recs = _json_records(RT / "extended" / "streaming_tta_study.json")
    if recs is None:
        print("[tab:streaming] missing"); return
    section("tab:streaming (first/last step per stream)")
    sname = {"feature_drift_stream": "Feature drift", "edge_perturb_stream": "Edge perturbation",
             "homophily_drift_stream": "Homophily drift"}
    label = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes", "full_method": "Full method"}
    for stream in ["feature_drift_stream", "edge_perturb_stream", "homophily_drift_stream"]:
        sub = [r for r in recs if r["stream"] == stream]
        if not sub:
            continue
        last = max(r["step"] for r in sub)
        for step in [0, last]:
            for m, lab in label.items():
                rs = [r for r in sub if r["step"] == step and r["method"] == m]
                if not rs:
                    continue
                inten = rs[0]["intensity"]
                trig = "--" if m != "full_method" else f"{100*mean(1.0 if r['detector_triggered'] else 0.0 for r in rs):.0f}"
                print(f"{sname[stream]} & {step+1} ({inten}) & {lab} & ${_ms([r['accuracy'] for r in rs])}$ & {trig} \\\\")
        print("\\cmidrule(lr){1-5}")


def tab_dual():
    recs = _json_records(RT / "extended" / "dual_checkpoint_streaming.json")
    stream = _json_records(RT / "extended" / "streaming_tta_study.json")
    if recs is None:
        print("[tab:dual-checkpoint] missing"); return
    section("tab:dual-checkpoint (final step, homophily 0.50)")
    pname = {"none": "None (adapt every step)", "single_checkpoint": "Single checkpoint (to previous step)",
             "dual_checkpoint": "Dual checkpoint (to source on 2 consecutive)"}
    final = [r for r in recs if abs(r["intensity"] - 0.50) < 1e-9]
    for policy in ["none", "single_checkpoint", "dual_checkpoint"]:
        rs = [r for r in final if r["policy"] == policy]
        if rs:
            print(f"{pname[policy]} & ${_ms([r['accuracy'] for r in rs])}$ & "
                  f"{100*mean(1.0 if r['triggered'] else 0.0 for r in rs):.0f} \\\\")
    if stream:
        src = [r for r in stream if r["stream"] == "homophily_drift_stream" and r["method"] == "source_only"
               and abs(r["intensity"] - 0.50) < 1e-9]
        if src:
            print(f"Source-only reference & ${_ms([r['accuracy'] for r in src])}$ & -- \\\\")


# ------------------------------------------------------------------ tab:adversarial
def tab_adversarial():
    recs = _json_records(RT / "extended" / "adversarial_study.json")
    if recs is None:
        print("[tab:adversarial] missing"); return
    section("tab:adversarial")
    aname = {"adv_edge_add": "Adv.\\ edge add", "adv_feature": "Adv.\\ feature"}
    label = {"source_only": "Source only", "entropy_all_nodes": "Entropy all nodes", "full_method": "Full method"}
    for attack, inten in [("adv_edge_add", 0.10), ("adv_edge_add", 0.20), ("adv_feature", 0.20), ("adv_feature", 0.40)]:
        cells = {}
        for m in label:
            rs = [r for r in recs if r["attack"] == attack and abs(r["intensity"] - inten) < 1e-9 and r["method"] == m]
            if rs:
                cells[m] = ([r["accuracy"] for r in rs], [r["ece"] for r in rs])
        if len(cells) < 3:
            continue
        src_ece = mean(cells["source_only"][1])
        best_acc = max(mean(v[0]) for v in cells.values())
        best_ece = min(mean(v[1]) for v in cells.values())
        deltas = {m: mean(v[1]) - src_ece for m, v in cells.items()}
        best_delta = min(d for m, d in deltas.items() if m != "source_only")
        for m, lab in label.items():
            accs, eces = cells[m]
            if m == "source_only":
                d = "--"
            else:
                dm = f"{deltas[m]:+.4f}"
                d = f"$\\mathbf{{{dm}}}$" if abs(deltas[m] - best_delta) < 5e-5 else f"${dm}$"
            print(f"{aname[attack]} ({inten:.2f}) & {lab} & "
                  f"${_ms(accs, bold=abs(mean(accs)-best_acc)<5e-5)}$ & "
                  f"${_ms(eces, bold=abs(mean(eces)-best_ece)<5e-5)}$ & {d} \\\\")
        print("\\cmidrule(lr){1-5}")


# ------------------------------------------------------------------ tab:scalability
def tab_scalability():
    rows = _csv_rows(RT / "scalability_summary.csv")
    if rows is None:
        print("[tab:scalability] missing"); return
    section("tab:scalability (mean adaptation runtime over 3 seeds x 3 shifts)")
    by = {(int(r["scale"]), r["method"]): float(r["mean_runtime"]) for r in rows}
    scales = sorted({int(r["scale"]) for r in rows})
    label = {"tent_entropy": "Tent (entropy only)", "eata_filter": "EATA (filtered entropy)",
             "full_method": "Full method"}
    for m, lab in label.items():
        vals = [by.get((s, m)) for s in scales]
        print(f"{lab} & " + " & ".join(f"${v:.2f}$" if v is not None else "n/a" for v in vals) + " \\\\")
    tent = [by.get((s, "tent_entropy")) for s in scales]
    full = [by.get((s, "full_method")) for s in scales]
    if all(tent) and all(full):
        print("Overhead vs.\\ Tent & " + " & ".join(f"${f/t:.1f}\\times$" for f, t in zip(full, tent)) + " \\\\")
    acc = {(int(r["scale"]), r["method"]): float(r["mean_accuracy"]) for r in rows}
    print("% accuracy check: " + ", ".join(f"n={s}: src={acc.get((s,'source_only'),0):.3f} full={acc.get((s,'full_method'),0):.3f}" for s in scales))


# ------------------------------------------------------------------ appendix tables
def app_detector():
    recs = _json_records(RT / "supplementary" / "detector_sensitivity.json")
    if recs is None:
        print("[app detector-sensitivity] missing"); return
    section("appendix tab:detector-sensitivity")
    cname = {("homophily_shift", 0.50): "Homophily shift (out-of-env)",
             ("homophily_shift", 0.25): "Homophily shift (boundary)",
             ("feature_noise", 0.45): "Feature noise (in-envelope)"}
    by = defaultdict(list)
    for r in recs:
        by[(r["shift"], float(r["intensity"]), r["delta_tolerance"], r["phi_tolerance"])].append(r)
    for key in sorted(by, key=lambda k: (k[0], -k[1], k[2])):
        shift, inten, dt, pt = key
        rs = by[key]
        trig = 100 * mean(1.0 if r["triggered"] else 0.0 for r in rs)
        print(f"{cname.get((shift, inten), f'{shift} {inten}')} & {inten:.2f} & {dt:.2f} & {pt:.2f} & "
              f"{trig:.1f} & ${_ms([r['final_accuracy'] for r in rs])}$ \\\\")


def app_threshold():
    recs = _json_records(RT / "extended" / "threshold_calibration.json")
    if recs is None:
        print("[app threshold-calib] missing"); return
    section("appendix tab:threshold-calib (Delta* = k x delta_self)")
    by_k = defaultdict(lambda: defaultdict(list))
    for r in recs:
        regime = "out" if (r["shift"] == "homophily_shift" and r["intensity"] >= 0.50) else "in"
        by_k[r["k"]][regime].append(r)
    for k in sorted(by_k):
        ie, oe = by_k[k]["in"], by_k[k]["out"]
        print(f"{k:.1f} & {100*mean(1.0 if r['triggered'] else 0.0 for r in ie):.1f} & "
              f"${_ms([r['accuracy'] for r in ie])}$ & "
              f"{100*mean(1.0 if r['triggered'] else 0.0 for r in oe):.1f} & "
              f"${_ms([r['accuracy'] for r in oe])}$ \\\\")
    print(f"% delta_self values: {sorted(set(round(r['delta_self'], 5) for r in recs))}")


def app_lambda():
    recs = _json_records(RT / "supplementary" / "lambda_sensitivity.json")
    if recs is None:
        print("[app lambda-sensitivity] missing"); return
    section("appendix tab:lambda-sensitivity")
    by = defaultdict(list)
    for r in recs:
        by[(r["lambda_cal"], r["lambda_af"])].append(r)
    for (lc, la) in sorted(by):
        rs = by[(lc, la)]
        print(f"{lc} & {la} & ${_ms([r['accuracy'] for r in rs])}$ & ${_ms([r['ece'] for r in rs])}$ \\\\")
    accs = [mean(r["accuracy"] for r in by[k]) for k in by]
    eces = [mean(r["ece"] for r in by[k]) for k in by]
    print(f"% spread across grid: acc {max(accs)-min(accs):.4f}, ece {max(eces)-min(eces):.4f}")


def app_drift():
    recs = _json_records(RT / "supplementary" / "drift_trajectory.json")
    if recs is None:
        print("[app subgroup-drift] missing"); return
    section("appendix tab:subgroup-drift (final step)")
    by = defaultdict(list)
    for r in recs:
        by[(r["shift"], float(r["intensity"]))].append(r)
    for (shift, inten), rs in sorted(by.items()):
        last = max(r["step"] for r in rs)
        fin = [r for r in rs if r["step"] == last]
        print(f"{_cond_label(shift, inten)} & ${mean(r['drift_low'] for r in fin):+.4f}$ & "
              f"${mean(r['drift_mid'] for r in fin):+.4f}$ & ${mean(r['drift_high'] for r in fin):+.4f}$ & "
              f"${mean(r['delta_t'] for r in fin):.4f}$ & ${mean(r['phi_t'] for r in fin):.4f}$ \\\\")


def app_corr():
    recs = _json_records(RT / "supplementary" / "component_correlation.json")
    if recs is None:
        print("[app signal-corr] missing"); return
    section("appendix tab:signal-corr (mean pairwise correlation)")
    by = defaultdict(list)
    for r in recs:
        by[(r["signal_a"], r["signal_b"])].append(r["correlation"])
    for (a, b), vals in sorted(by.items()):
        print(f"{a} & {b} & ${mean(vals):+.4f}$ \\\\")


ALL = {
    "main": tab_main, "prose": prose_main, "ablation": tab_ablation, "public": tab_public,
    "fair": tab_fair, "significance": tab_significance, "assumption": tab_assumption,
    "backbone": tab_backbone, "large": tab_large, "webkb": tab_webkb,
    "webkb_assumption": tab_webkb_assumption, "boundary": tab_boundary,
    "streaming": tab_streaming, "dual": tab_dual, "adversarial": tab_adversarial,
    "scalability": tab_scalability, "app_detector": app_detector, "app_threshold": app_threshold,
    "app_lambda": app_lambda, "app_drift": app_drift, "app_corr": app_corr,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(ALL)
    for t in targets:
        ALL[t]()
