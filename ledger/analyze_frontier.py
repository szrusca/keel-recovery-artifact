import argparse
import glob
import json
import os
import time
from collections import defaultdict


def load_rows(paths):
    rows = []
    config = None
    for p in paths:
        d = json.load(open(p))
        config = config or d["config"]
        rows.extend(d["rows"])
    return config, rows


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    out = {}
    for policy, rs in by.items():
        n = len(rs)
        succ = sum(1 for r in rs if r["success"])
        usd = sum(r["usd"] for r in rs)
        out[policy] = {
            "n": n,
            "success_rate": succ / n,
            "progress_mean": sum(r["progress"] for r in rs) / n,
            "local_progress_mean": sum(r.get("local_progress", 0.0) for r in rs) / n,
            "usd_per_task": usd / n,
            "usd_per_success": (usd / succ) if succ else None,
            "calls_per_task": sum(r["calls"] for r in rs) / n,
            "tokens_per_task": sum(r["tokens"] for r in rs) / n,
            "retries_per_task": sum(r["retries"] for r in rs) / n,
            "rollbacks_per_task": sum(r["rollbacks"] for r in rs) / n,
            "escalated_share": sum(1 for r in rs if r["escalated"]) / n,
            "escalations_per_task": sum(r.get("escalations", 0) for r in rs) / n,
            "aborted_share": sum(1 for r in rs if r["aborted"]) / n,
        }
    return out


def pareto(summary):
    pts = [(p, v["usd_per_task"], v["success_rate"]) for p, v in summary.items()]
    front = []
    for p, c, s in pts:
        dominated = any((c2 <= c and s2 >= s) and (c2 < c or s2 > s) for _, c2, s2 in pts)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: summary[p]["usd_per_task"])


def render(title, config, summary, front, paths):
    lines = [f"# {title}", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ", f"Data files: {', '.join(os.path.basename(p) for p in paths)}  "]
    if config:
        lines.append(f"Base model: {config['provider']} / {config['model']}, escalation target: {config['switch_provider']} / {config['switch_model']}, verifier: {config['verifier']}, K={config['k']}, checkpoint every {config['checkpoint_every']}, budget cap {config['usd_cap']} USD, steps {config['n_steps']}, ops per step {config['ops']} ")
    lines += ["", "## Policies", "", "| Policy | n | Task success | Global progress | Local progress | USD/task | USD/success | Calls/task | Tokens/task | Retries/task | Rollbacks/task | Escalated share | Budget-cut share |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, v in sorted(summary.items(), key=lambda kv: kv[1]["usd_per_task"]):
        ups = "-" if v["usd_per_success"] is None else f"{v['usd_per_success']:.5f}"
        lines.append(f"| {p} | {v['n']} | {100 * v['success_rate']:.1f}% | {100 * v['progress_mean']:.1f}% | {100 * v['local_progress_mean']:.1f}% | {v['usd_per_task']:.5f} | {ups} | {v['calls_per_task']:.1f} | {v['tokens_per_task']:.0f} | {v['retries_per_task']:.2f} | {v['rollbacks_per_task']:.2f} | {100 * v['escalated_share']:.0f}% | {100 * v['aborted_share']:.0f}% |")
    lines += ["", "## Pareto frontier (USD/task vs task success)", "", "Cheapest first: " + " → ".join(front), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--title", default="Cost frontier of recovery policies")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p))
    config, rows = load_rows(paths)
    summary = summarize(rows)
    front = pareto(summary)
    md = render(args.title, config, summary, front, paths)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(md)
    json.dump({"summary": summary, "pareto": front}, open(args.out.replace(".md", ".json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
