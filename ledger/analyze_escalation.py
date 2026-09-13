import argparse
import glob
import json
from collections import defaultdict

from .events import read_events


def summarize(paths, base_provider):
    events = []
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            events.extend(read_events(p))
    tasks = defaultdict(lambda: {"calls": [], "end": None})
    for e in events:
        key = (e["run_id"], e["seed"], e["policy"])
        if e["kind"] == "ModelResponded":
            tasks[key]["calls"].append(e)
        elif e["kind"] == "TaskEnded":
            tasks[key]["end"] = e
    latest = {}
    for key, t in tasks.items():
        if t["end"] is None:
            continue
        prev = latest.get((key[1], key[2]))
        if prev is None or t["end"]["ts"] > prev["end"]["ts"]:
            latest[(key[1], key[2])] = t
    by_policy = defaultdict(list)
    for (seed, policy), t in latest.items():
        key = (None, seed, policy)
        calls = t["calls"]
        target = [c for c in calls if c["provider"] != base_provider]
        answered = [c for c in target if c["status"] == 200]
        usd_total = sum(c.get("usd", 0.0) for c in calls)
        usd_target = sum(c.get("usd", 0.0) for c in target)
        tok_total = sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) + c.get("thought_tokens", 0) for c in calls)
        tok_target = sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) + c.get("thought_tokens", 0) for c in target)
        by_policy[key[2]].append({
            "seed": key[1],
            "success": bool(t["end"]["success"]),
            "usd": usd_total,
            "latency_s": sum(c.get("latency_ms", 0) for c in calls) / 1000.0,
            "calls": len(calls),
            "base_calls": len(calls) - len(target),
            "target_answered": len(answered),
            "target_rejected": len(target) - len(answered),
            "target_cost_share": usd_target / usd_total if usd_total else 0.0,
            "target_token_share": tok_target / tok_total if tok_total else 0.0,
            "retries": t["end"].get("retries", 0),
            "escalations": t["end"].get("escalations", 0),
            "local_progress": t["end"].get("local_progress", 0.0),
        })
    return by_policy


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows) if rows else 0.0


def render(by_policy, label):
    lines = [f"| Policy | Success | m$/task | Latency s/task | Calls/task (base + target answered + target rejected) | Target cost share | Target token share | Retries/task | Escalations/task | Local progress |", "|---|---|---|---|---|---|---|---|---|---|"]
    for policy, rows in sorted(by_policy.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        ok = sum(1 for r in rows if r["success"])
        lines.append(f"| {label} {policy} | {ok}/{len(rows)} | {1000 * mean(rows, 'usd'):.2f} | {mean(rows, 'latency_s'):.0f} | {mean(rows, 'calls'):.1f} = {mean(rows, 'base_calls'):.1f} + {mean(rows, 'target_answered'):.1f} + {mean(rows, 'target_rejected'):.1f} | {100 * mean(rows, 'target_cost_share'):.0f}% | {100 * mean(rows, 'target_token_share'):.0f}% | {mean(rows, 'retries'):.2f} | {mean(rows, 'escalations'):.1f} | {100 * mean(rows, 'local_progress'):.0f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--base-provider", default="mistral")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    by_policy = summarize(args.paths, args.base_provider)
    print(render(by_policy, args.label))
    if args.json_out:
        json.dump(by_policy, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
