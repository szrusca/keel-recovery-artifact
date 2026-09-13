import argparse
import glob
import json
import math
import os
from collections import defaultdict

from .events import read_events

STRATEGIES = ["same", "hot", "rephrase", "reset", "switch"]


def load_points(paths, semantic_only=True):
    events = []
    for p in paths:
        events.extend(read_events(p))
    main = {(e["seed"], e["t"]): e for e in events if e["kind"] == "ModelResponded" and e.get("path") == "main"}
    p_step = sum(1 for e in main.values() if e["outcome"] == "ok") / len(main)
    points = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e["kind"] == "ModelResponded" and e.get("path") == "retry":
            points[(e["seed"], e["t"])][e["strategy"]].append((e["attempt"], e["outcome"]))
    keep = {}
    for key, strat in points.items():
        m = main.get(key)
        if m is None or m["outcome"] == "ok":
            continue
        if semantic_only and m["outcome"] != "semantic":
            continue
        keep[key] = {s: [o for _, o in sorted(v)] for s, v in strat.items()}
    return p_step, keep


def first_success(outcomes):
    for j, o in enumerate(outcomes, 1):
        if o == "ok":
            return j
    return None


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_geometric_loglik(a, b, firsts, kmax):
    ll = 0.0
    lb = log_beta(a, b)
    for j in firsts:
        if j is None:
            ll += log_beta(a, b + kmax) - lb
        else:
            ll += log_beta(a + 1, b + j - 1) - lb
    return ll


def fit_beta_geometric(firsts, kmax):
    best = None
    grid = [math.exp(x) for x in [i / 8.0 for i in range(-32, 33)]]
    for a in grid:
        for b in grid:
            ll = beta_geometric_loglik(a, b, firsts, kmax)
            if best is None or ll > best[0]:
                best = (ll, a, b)
    return best


def beta_cdf(x, a, b, n=2000):
    total = 0.0
    lb = log_beta(a, b)
    for i in range(n):
        u = (i + 0.5) / n * x
        total += math.exp((a - 1) * math.log(u) + (b - 1) * math.log(1 - u) - lb)
    return total * x / n


def predicted_recovery(a, b, k):
    return 1.0 - math.exp(log_beta(a, b + k) - log_beta(a, b))


def analyze(p_step, points, kmax):
    n = len(points)
    out = {"n_points": n, "p_step": p_step, "kmax": kmax}
    out["naive_independent"] = [1 - (1 - p_step) ** k for k in range(1, kmax + 1)]
    per_strategy = {}
    for s in STRATEGIES:
        firsts = [first_success(v[s]) for v in points.values() if s in v]
        if not firsts:
            continue
        rec = [sum(1 for j in firsts if j is not None and j <= k) / len(firsts) for k in range(1, kmax + 1)]
        per_strategy[s] = {"n": len(firsts), "recovered_within_k": rec, "never": sum(1 for j in firsts if j is None) / len(firsts)}
    out["per_strategy"] = per_strategy
    firsts_same = [first_success(v["same"]) for v in points.values() if "same" in v]
    ll, a, b = fit_beta_geometric(firsts_same, kmax)
    out["beta_geometric_same"] = {"alpha": a, "beta": b, "loglik": ll, "mean_conditional_p": a / (a + b), "share_p_below_0.1": beta_cdf(0.1, a, b), "share_p_below_0.05": beta_cdf(0.05, a, b), "predicted_recovery": [predicted_recovery(a, b, k) for k in range(1, kmax + 1)]}
    same_model = ["same", "hot", "rephrase", "reset"]
    dead_all = 0
    counted = 0
    total_attempts = 0
    for v in points.values():
        if all(s in v for s in same_model):
            counted += 1
            total_attempts += sum(len(v[s]) for s in same_model)
            if all(first_success(v[s]) is None for s in same_model):
                dead_all += 1
    out["dead_under_all_same_model"] = {"n": counted, "share": dead_all / counted if counted else None, "attempts_per_point": total_attempts / counted if counted else None}
    return out


def render(title, groups):
    lines = [f"# {title}", ""]
    lines.append("Reviewer point M1: the independent-failure model evaluated at the marginal p is not the right null for the failure-conditioned sample. Below, the same-prompt retry sequences (geometric, censored) are fitted with a Beta-geometric model to estimate the distribution of conditional success probability over failure points, and the share of failure points that none of the 16 same-model attempts across four perturbations recovered is reported.")
    lines.append("")
    lines.append("| Group | Failure points | Marginal p | Independent-model R(4) | Same-prompt R(4) | Beta-geometric mean p | Fitted-null R(4) | P(p_i<0.1) | Dead after 16 same-model attempts | hot / rephrase / reset R(4) | switch R(4) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, a in groups:
        bg = a["beta_geometric_same"]
        ps = a["per_strategy"]
        k = a["kmax"]
        lines.append(f"| {name} | {a['n_points']} | {a['p_step']:.2f} | {100*a['naive_independent'][k-1]:.0f}% | {100*ps['same']['recovered_within_k'][k-1]:.0f}% | {bg['mean_conditional_p']:.2f} | {100*bg['predicted_recovery'][k-1]:.0f}% | {100*bg['share_p_below_0.1']:.0f}% | {100*a['dead_under_all_same_model']['share']:.0f}% | {100*ps['hot']['recovered_within_k'][k-1]:.0f} / {100*ps['rephrase']['recovered_within_k'][k-1]:.0f} / {100*ps['reset']['recovered_within_k'][k-1]:.0f} | {100*ps['switch']['recovered_within_k'][k-1]:.0f}% |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- The marginal p is the success rate over all main-path steps; failure points are the subset conditioned on failure, so their conditional success is necessarily lower. The independent prediction 1-(1-p)^4 uses the marginal p and is therefore not the null for this sample.")
    lines.append("- The Beta-geometric fit models each failure point's conditional success p_i as Beta-distributed and maximizes the likelihood of the first-success positions of the same-prompt retries. Agreement between the fitted R(4) and the same-prompt R(4) is by construction; the informative quantity is the shape of the fitted distribution, and P(p_i<0.1) is the share of near-deterministic errors.")
    lines.append("- If the R(4) of hot, rephrase, and reset fall near the same-prompt fitted prediction, those perturbations are further independent draws over the same heterogeneous items and none of them changes which items are solvable.")
    lines.append("- The share of failure points dead after 16 same-model attempts needs no model: these points are (near-)deterministic errors for that model.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmax", type=int, default=4)
    args = ap.parse_args()
    groups_spec = [
        ("ministral-8b", ["runs/pilot/mistral_ministral-8b-latest_20260909-065452.events.jsonl"]),
        ("ministral-14b", ["runs/pilot/mistral_ministral-14b-latest_20260907-215521.events.jsonl"]),
        ("ministral-14b, T=0", ["runs/pilot/mistral_ministral-14b-latest_20260909-065452.events.jsonl"]),
        ("nemotron reasoning off (K=4 round)", ["runs/pilot/nvidia_nvidia-nemotron-3-super-120b-a12b_r-none_20260909-080447.events.jsonl"]),
        ("Gemini Flash-Lite (round 2)", ["runs/pilot/gemini_gemini-3.5-flash-lite_20260909-080447.events.jsonl"]),
    ]
    groups = []
    for name, paths in groups_spec:
        paths = [p for pat in paths for p in glob.glob(pat)]
        p_step, points = load_points(paths)
        a = analyze(p_step, points, args.kmax)
        groups.append((name, a))
    md = render("Experiment 1 supplement: difficulty heterogeneity of failure points and the corrected independence null", groups)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(md)
    json.dump({n: a for n, a in groups}, open(args.out.replace(".md", ".json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
