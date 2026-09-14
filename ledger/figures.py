import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})

SERIES = {"same": "#2a78d6", "hot": "#eb6834", "rephrase": "#1baf7a", "reset": "#eda100", "switch": "#e87ba4"}
POLICY_COLORS = {"none": "#52514e", "retry": "#2a78d6", "reset": "#eda100", "rollback": "#1baf7a", "budget": "#eb6834", "escalate": "#e87ba4"}
SERIES_MARKERS = {"same": "o", "hot": "s", "rephrase": "^", "reset": "D", "switch": "v"}
POLICY_MARKERS = {"none": "o", "retry": "s", "reset": "^", "rollback": "D", "budget": "v", "escalate": "*"}
SERIES_STYLES = {"same": "-", "hot": (0, (5, 1.5)), "rephrase": (0, (1, 1.2)), "reset": (0, (4, 1.2, 1, 1.2)), "switch": "-"}
LABELS = {"same": "same prompt", "hot": "temperature 1.0", "rephrase": "rephrased prompt", "reset": "cleared context", "switch": "switch to reasoning-on target"}
POLICY_LABELS = {"none": "none", "retry": "retry", "reset": "reset", "rollback": "rollback", "budget": "budget", "escalate": "escalate"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def fig_recovery(panels, out, hetero_json):
    H = json.load(open(hetero_json))
    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False, sharey=True)
    for ax, (title, key) in zip(axes.flat, panels):
        a = H[key]
        kmax = a["kmax"]
        ks = list(range(1, kmax + 1))
        ax.plot(ks, [100 * v for v in a["naive_independent"]], color=MUTED, linestyle="--", linewidth=1.6, label="s-independent failure model, marginal p, (1)")
        ax.plot(ks, [100 * v for v in a["beta_geometric_same"]["predicted_recovery"]], color=MUTED, linestyle=":", linewidth=1.8, label="heterogeneity null fitted to same-prompt retries, (2)")
        for s_, color in SERIES.items():
            if s_ in a["per_strategy"]:
                ax.plot(ks, [100 * v for v in a["per_strategy"][s_]["recovered_within_k"]], color=color, linewidth=1.7, linestyle=SERIES_STYLES[s_], marker=SERIES_MARKERS[s_], markersize=5, markeredgecolor="white", markeredgewidth=0.7, label=LABELS[s_])
        ax.set_title(f"{title}  (p={a['p_step']:.2f}, n={a['n_points']})", fontsize=9, loc="left")
        ax.set_xticks(ks)
        ax.set_ylim(0, 102)
        ax.set_xlabel("retries k", fontsize=8)
        style(ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel("semantic failure points recovered within k (%)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


EXTRA_POLICIES = {"escalate_step": ("#4a3aa7", "escalate-and-return"), "target_only": ("#008300", "target only")}
EXTRA_MARKERS = {"escalate_step": "P", "target_only": "X"}


def extra_points(extra):
    pts = []
    for label, pattern in extra:
        rows = []
        for path in sorted(glob.glob(pattern)):
            rows += json.load(open(path))["rows"]
        if not rows:
            continue
        pts.append((label, 1000 * sum(r["usd"] for r in rows) / len(rows), 100 * sum(1 for r in rows if r["success"]) / len(rows)))
    return pts


def fig_frontier(cells, out, extras=None):
    cols = 2 if len(cells) > 2 else len(cells)
    rows = (len(cells) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.45 * cols, 2.9 * rows), squeeze=False, sharey=True)
    flat = list(axes.flat)
    for idx, (ax, (title, path)) in enumerate(zip(flat, cells)):
        d = json.load(open(path))
        frontier_pts = []
        if extras and idx in extras:
            for label, x, y in extra_points(extras[idx]):
                color, name = EXTRA_POLICIES[label]
                ax.scatter([x], [y], s=64, color=color, marker=EXTRA_MARKERS[label], edgecolors="white", linewidths=1.0, zorder=3, label=name)
                if label == "target_only":
                    ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, -12), fontsize=7.5, color=TEXT)
                else:
                    ax.annotate(name, (x, y), textcoords="offset points", xytext=(-4, 7), fontsize=7.5, color=TEXT, ha="right")
                frontier_pts.append((x, y))
        for p in ["none", "reset", "rollback", "budget", "retry", "escalate"]:
            if p not in d["summary"]:
                continue
            v = d["summary"][p]
            x = 1000 * v["usd_per_task"]
            y = 100 * v["success_rate"]
            ax.scatter([x], [y], s=110 if p == "escalate" else 52, color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], edgecolors="white", linewidths=1.0, zorder=3, label=POLICY_LABELS[p])
            frontier_pts.append((x, y if round(v["success_rate"] * v["n"]) > 1 else 0.0))
            if p in ("escalate", "retry"):
                ax.annotate(POLICY_LABELS[p], (x, y), textcoords="offset points", xytext=(-4, 7), fontsize=7.5, color=TEXT, ha="right" if p == "escalate" else "left")
        front = []
        for x, y in frontier_pts:
            if not any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for x2, y2 in frontier_pts):
                front.append((x, y))
        front.sort()
        ax.step([q[0] for q in front], [q[1] for q in front], where="post", color=MUTED, linewidth=1, linestyle=":", zorder=2)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("cost per task (m$, list price)", fontsize=8)
        ax.set_ylim(-5, 108)
        style(ax)
    for ax in flat[len(cells):]:
        ax.axis("off")
    for r in range(rows):
        axes[r][0].set_ylabel("task success rate (%)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def pooled_escalate(patterns):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            rows += [r for r in json.load(open(path))["rows"] if r["policy"] == "escalate"]
    succ = sum(1 for r in rows if r["success"])
    usd = sum(r["usd"] for r in rows)
    return 100 * succ / len(rows), (1000 * usd / succ) if succ else 0.0, len(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def fig_verifier(cells, out, pooled=None):
    names = [c[0] for c in cells]
    succ = []
    cps = []
    lo = []
    hi = []
    for name, path in cells:
        if pooled and name in pooled:
            sr, cp, n = pooled_escalate(pooled[name])
            k = round(sr * n / 100)
        else:
            d = json.load(open(path))["summary"]["escalate"]
            sr = 100 * d["success_rate"]
            cp = 1000 * d["usd_per_success"] if d["usd_per_success"] else 0.0
            n = d["n"]
            k = round(d["success_rate"] * n)
        succ.append(sr)
        cps.append(cp)
        l, h = wilson(k, n)
        lo.append(sr - l)
        hi.append(h - sr)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].bar(names, succ, color="#e87ba4", width=0.45, yerr=[lo, hi], capsize=3, ecolor=TEXT, error_kw={"linewidth": 1})
    axes[0].set_title("escalate: task success by verifier (95% Wilson intervals)", fontsize=9, loc="left")
    axes[0].set_ylim(0, 112)
    for i, v in enumerate(succ):
        axes[0].text(i + 0.26, v + 1, f"{v:.0f}%", ha="left", fontsize=8, color=TEXT)
    axes[1].bar(names, cps, color="#2a78d6", width=0.45)
    axes[1].set_title("escalate: cost per success (m$)", fontsize=9, loc="left")
    for i, v in enumerate(cps):
        axes[1].text(i, v + 0.3, f"{v:.1f}" if v else "inf", ha="center", fontsize=8, color=TEXT)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_horizon(cells, out):
    steps = [c[0] for c in cells]
    none_cost = []
    esc_cost = []
    for _, path in cells:
        s = json.load(open(path))["summary"]
        none_cost.append(1000 * s["none"]["usd_per_task"])
        esc_cost.append(1000 * s["escalate"]["usd_per_task"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = range(len(steps))
    w = 0.36
    ax.bar([i - w / 2 for i in x], none_cost, width=w, color=POLICY_COLORS["none"], label="none (0% success)")
    ax.bar([i + w / 2 for i in x], esc_cost, width=w, color=POLICY_COLORS["escalate"], label="escalate (100% success)")
    for i, (a, b) in enumerate(zip(none_cost, esc_cost)):
        ax.text(i + w / 2, b + 0.4, f"{b / a:.1f}x", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s} steps" for s in steps])
    ax.set_ylabel("cost per task (m$)", fontsize=8)
    ax.set_title("cost per task vs. task length (ministral-8b, oracle verifier)", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="../paper-recovery/figures")
    args = ap.parse_args()
    R = args.results
    os.makedirs(args.out, exist_ok=True)
    fig_recovery([
        ("ministral-8b", "ministral-8b"),
        ("ministral-14b", "ministral-14b"),
        ("ministral-14b, T=0", "ministral-14b, T=0"),
        ("nemotron-3-super-120b, reasoning off", "nemotron reasoning off (K=4 round)"),
        ("Gemini 3.5 Flash-Lite", "Gemini Flash-Lite (round 2)"),
    ], f"{args.out}/fig1_recovery_gap.pdf", f"{R}/19-heterogeneity.json")
    fig_frontier([
        ("ministral-8b, 16 steps", f"{R}/06b-frontier-ministral8b.json"),
        ("ministral-14b, 16 steps", f"{R}/06d-frontier-ministral14b.json"),
        ("nemotron-3-super-120b reasoning off, 16 steps", f"{R}/06-frontier-nemotron.json"),
        ("Gemini 3.5 Flash-Lite, 16 steps", f"{R}/06e-frontier-gemini.json"),
    ], f"{args.out}/fig2_cost_frontier.pdf", extras={0: [("escalate_step", "runs/frontier_escstep_k0/*.rows.json"), ("target_only", "runs/frontier_target/*.rows.json")]})
    fig_verifier([
        ("oracle", f"{R}/06b-frontier-ministral8b.json"),
        ("noisy (20% FN), 4 seeds", f"{R}/06c-frontier-ministral8b-noisy.json"),
        ("format only", f"{R}/06f-frontier-ministral8b-format.json"),
    ], f"{args.out}/fig3_verifier.pdf", pooled={"noisy (20% FN), 4 seeds": ["runs/frontier/mistral_ministral-8b-latest_noisy_*.rows.json", "runs/frontier_noisy_seeds/*.rows.json"]})
    fig_horizon([
        (8, f"{R}/06g-frontier-ministral8b-h8.json"),
        (16, f"{R}/06b-frontier-ministral8b.json"),
        (32, f"{R}/06h-frontier-ministral8b-h32.json"),
    ], f"{args.out}/fig4_horizon.pdf")
    for f in sorted(glob.glob(f"{args.out}/*.pdf")):
        print("wrote", f)


if __name__ == "__main__":
    main()
