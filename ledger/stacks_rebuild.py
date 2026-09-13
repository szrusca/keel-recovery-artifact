import argparse
import json

from . import tasks_t1
from .events import read_events
from .stacks import sample_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--ops", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=7000)
    args = ap.parse_args()
    steps = sample_steps(args.n, args.seed_start, args.n_steps, args.ops, args.accounts)
    rows = {}
    stacks = []
    for e in read_events(args.events):
        if e["kind"] != "ModelResponded":
            continue
        idx = e["idx"]
        seed, step, _ = steps[idx]
        assert seed == e["seed"] and step["t"] == e["t"]
        row = rows.setdefault(idx, {"idx": idx, "seed": seed, "t": step["t"], "outputs": {}})
        kind, obj = tasks_t1.verify(e.get("text") or "", step["after"])
        if e.get("error_kind"):
            kind = e["error_kind"]
        row["outputs"][e["stack"]] = {"status": e["status"], "error": e.get("error_kind") or "", "kind": kind, "ledger": obj, "text": (e.get("text") or "").strip(), "latency_ms": e["latency_ms"], "tokens": e.get("input_tokens", 0) + e.get("output_tokens", 0) + e.get("thought_tokens", 0)}
        if e["stack"] not in stacks:
            stacks.append(e["stack"])
    complete = [r for _, r in sorted(rows.items()) if len(r["outputs"]) == len(stacks)]
    json.dump({"stacks": stacks, "results": complete}, open(args.out, "w"), indent=1)
    print("rebuilt", len(complete), "complete steps of", len(rows), "into", args.out)


if __name__ == "__main__":
    main()
