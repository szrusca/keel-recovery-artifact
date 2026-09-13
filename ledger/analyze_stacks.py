import argparse
import itertools
import json
from collections import Counter


def load(paths):
    stacks = []
    merged = {}
    for path in paths:
        d = json.load(open(path))
        for st in d["stacks"]:
            if st not in stacks:
                stacks.append(st)
        for r in d["results"]:
            merged[(r["seed"], r["t"])] = r
    return stacks, [merged[k] for k in sorted(merged)]


def answered(o):
    return o["status"] == 200 and not o.get("error")


def per_stack(stacks, results):
    lines = ["| Stack | Answered | Correct | Semantic errors | Format errors | Unanswered (infrastructure) | Median latency | Mean tokens |", "|---|---|---|---|---|---|---|---|"]
    for s in stacks:
        outs = [r["outputs"][s] for r in results if s in r["outputs"]]
        good = [o for o in outs if answered(o)]
        kinds = Counter(o["kind"] for o in good)
        lat = sorted(o["latency_ms"] for o in good) or [0]
        toks = [o.get("tokens", 0) for o in good]
        fmt = len(good) - kinds.get("ok", 0) - kinds.get("semantic", 0)
        lines.append(f"| {s} | {len(good)}/{len(outs)} | {kinds.get('ok', 0)} ({100 * kinds.get('ok', 0) / max(1, len(good)):.0f}%) | {kinds.get('semantic', 0)} | {fmt} | {len(outs) - len(good)} | {lat[len(lat) // 2] / 1000:.1f} s | {sum(toks) / max(1, len(toks)):.0f} |")
    return "\n".join(lines)


def pairwise(stacks, results):
    lines = ["| Stack pair | Steps | Same text | Same state | Exactly one side correct (left / right) |", "|---|---|---|---|---|"]
    for a, b in itertools.combinations(stacks, 2):
        rows = [r for r in results if a in r["outputs"] and b in r["outputs"] and answered(r["outputs"][a]) and answered(r["outputs"][b])]
        text_eq = sum(1 for r in rows if r["outputs"][a]["text"] == r["outputs"][b]["text"])
        sem_eq = sum(1 for r in rows if r["outputs"][a]["ledger"] is not None and r["outputs"][a]["ledger"] == r["outputs"][b]["ledger"])
        left = sum(1 for r in rows if r["outputs"][a]["kind"] == "ok" and r["outputs"][b]["kind"] != "ok")
        right = sum(1 for r in rows if r["outputs"][b]["kind"] == "ok" and r["outputs"][a]["kind"] != "ok")
        n = len(rows)
        lines.append(f"| {a} vs {b} | {n} | {text_eq}/{n} | {sem_eq}/{n} ({100 * sem_eq / max(1, n):.0f}%) | {left + right} ({left} / {right}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    stacks, results = load(args.paths)
    print(per_stack(stacks, results))
    print()
    print(pairwise(stacks, results))


if __name__ == "__main__":
    main()
