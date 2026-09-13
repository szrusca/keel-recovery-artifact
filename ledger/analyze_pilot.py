import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from .events import read_events


def failure_class(outcome):
    if outcome == "ok":
        return "ok"
    if outcome.startswith("infra"):
        return "infra"
    if outcome.startswith("format"):
        return "format"
    if outcome == "semantic":
        return "semantic"
    return "client"


def load(paths):
    events = []
    for p in paths:
        events.extend(read_events(p))
    return events


def analyze(events):
    main = [e for e in events if e["kind"] == "ModelResponded" and e.get("path") == "main"]
    retries = [e for e in events if e["kind"] == "ModelResponded" and e.get("path") == "retry"]
    n_main = len(main)
    n_ok = sum(1 for e in main if e["outcome"] == "ok")
    p = n_ok / n_main if n_main else 0.0

    by_t = defaultdict(lambda: [0, 0])
    for e in main:
        by_t[e["t"]][0] += 1
        by_t[e["t"]][1] += e["outcome"] == "ok"

    fail_types = Counter(failure_class(e["outcome"]) for e in main if e["outcome"] != "ok")
    fail_detail = Counter(e["outcome"] for e in main if e["outcome"] != "ok")

    points = defaultdict(dict)
    for e in retries:
        key = (e["seed"], e["t"])
        s = e["strategy"]
        points[key].setdefault(s, []).append((e["attempt"], e["outcome"], e["input_tokens"], e["output_tokens"], e["thought_tokens"], e["usd"], e["latency_ms"]))
    main_class = {(e["seed"], e["t"]): failure_class(e["outcome"]) for e in main if e["outcome"] != "ok"}

    strategies = sorted({e["strategy"] for e in retries})
    kmax = max((e["attempt"] for e in retries), default=0)

    def curve(subset_keys):
        out = {}
        for s in strategies:
            n = 0
            rec = [0] * (kmax + 1)
            first_attempt_fail = 0
            first_attempt_n = 0
            tokens = 0
            usd = 0.0
            for key in subset_keys:
                attempts = sorted(points.get(key, {}).get(s, []))
                if not attempts:
                    continue
                n += 1
                first_attempt_n += 1
                first_attempt_fail += attempts[0][1] != "ok"
                for a in attempts:
                    tokens += a[2] + a[3] + a[4]
                    usd += a[5]
                hit = next((a[0] for a in attempts if a[1] == "ok"), None)
                if hit is not None:
                    for k in range(hit, kmax + 1):
                        rec[k] += 1
            out[s] = {
                "n": n,
                "recovered_within_k": [rec[k] / n if n else 0.0 for k in range(1, kmax + 1)],
                "eps1": first_attempt_fail / first_attempt_n if first_attempt_n else None,
                "tokens_per_point": tokens / n if n else 0.0,
                "usd_per_point": usd / n if n else 0.0,
                "usd_per_recovery": (usd / rec[kmax]) if rec[kmax] else None,
            }
        return out

    all_keys = list(main_class)
    overall = curve(all_keys)
    by_class = {c: curve([k for k, v in main_class.items() if v == c]) for c in sorted(set(main_class.values()))}
    theory = [1 - (1 - p) ** k for k in range(1, kmax + 1)]

    main_tokens = sum(e["input_tokens"] + e["output_tokens"] + e["thought_tokens"] for e in main)
    main_usd = sum(e["usd"] for e in main)
    return {
        "n_main_steps": n_main,
        "p_step": p,
        "eps0": 1 - p,
        "p_by_t": {t: v[1] / v[0] for t, v in sorted(by_t.items())},
        "failure_classes": dict(fail_types),
        "failure_detail": dict(fail_detail),
        "n_failure_points": len(all_keys),
        "kmax": kmax,
        "theory_independent": theory,
        "overall": overall,
        "by_class": by_class,
        "main_tokens": main_tokens,
        "main_usd": main_usd,
        "n_retry_calls": len(retries),
        "retry_usd": sum(e["usd"] for e in retries),
        "retry_latency_ms_mean": (sum(e["latency_ms"] for e in retries) / len(retries)) if retries else 0,
    }


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def render_md(title, meta, a):
    lines = [f"# {title}", ""]
    lines.append(f"Generated: {meta['generated']}  ")
    lines.append(f"Event files: {', '.join(os.path.basename(p) for p in meta['paths'])}  ")
    if meta.get("config"):
        c = meta["config"]
        lines.append(f"Base model: {c['provider']} / {c['model']}, switch target: {c['switch_provider']} / {c['switch_model']}, temperature {c['temperature']}, K={c['k']}, steps {c['n_steps']}, ops per step {c['ops_per_step']}, accounts {c['n_accounts']}  ")
    lines.append("")
    lines.append("## Main path")
    lines.append("")
    lines.append(f"- Main-path steps: {a['n_main_steps']}, per-step success p = {a['p_step']:.3f}, base error rate eps0 = {a['eps0']:.3f}")
    lines.append(f"- Failure points: {a['n_failure_points']}, by class: {a['failure_classes']}")
    lines.append(f"- Failure detail: {a['failure_detail']}")
    lines.append(f"- Main-path tokens: {a['main_tokens']}, list-price USD: {a['main_usd']:.4f}")
    lines.append(f"- Retry calls: {a['n_retry_calls']}, list-price USD: {a['retry_usd']:.4f}, mean latency {a['retry_latency_ms_mean']:.0f} ms")
    lines.append("")
    lines.append("### Success by step depth")
    lines.append("")
    lines.append("| t | p(t) |")
    lines.append("|---|---|")
    for t, v in a["p_by_t"].items():
        lines.append(f"| {t} | {v:.3f} |")
    lines.append("")
    kmax = a["kmax"]
    lines.append("## Retry-independence gap: recovery within k attempts")
    lines.append("")
    header = "| Strategy | n | " + " | ".join(f"k={k}" for k in range(1, kmax + 1)) + " | eps1 | eps1/eps0 | tokens/failure point | USD/failure point | USD/recovery |"
    lines.append(header)
    lines.append("|" + "---|" * (kmax + 7))
    lines.append("| Independent-failure model | - | " + " | ".join(fmt_pct(x) for x in a["theory_independent"]) + " | " + fmt_pct(a["eps0"]) + " | 1.00 | - | - | - |")
    for s, v in a["overall"].items():
        ratio = (v["eps1"] / a["eps0"]) if (v["eps1"] is not None and a["eps0"] > 0) else None
        lines.append(
            f"| {s} | {v['n']} | " + " | ".join(fmt_pct(x) for x in v["recovered_within_k"]) +
            f" | {fmt_pct(v['eps1'])} | {'-' if ratio is None else f'{ratio:.2f}'} | {v['tokens_per_point']:.0f} | {v['usd_per_point']:.5f} | {'-' if v['usd_per_recovery'] is None else f'{v['usd_per_recovery']:.5f}'} |"
        )
    lines.append("")
    lines.append("## By failure class")
    lines.append("")
    for cls, table in a["by_class"].items():
        lines.append(f"### {cls}")
        lines.append("")
        lines.append("| Strategy | n | " + " | ".join(f"k={k}" for k in range(1, kmax + 1)) + " | eps1 |")
        lines.append("|" + "---|" * (kmax + 3))
        for s, v in table.items():
            lines.append(f"| {s} | {v['n']} | " + " | ".join(fmt_pct(x) for x in v["recovered_within_k"]) + f" | {fmt_pct(v['eps1'])} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--title", default="Retry-independence gap pilot results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p))
    events = load(paths)
    a = analyze(events)
    config = None
    for p in paths:
        cand = p.replace(".events.jsonl", ".summary.json")
        if os.path.exists(cand):
            config = json.load(open(cand))["config"]
            break
    import time
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "paths": paths, "config": config}
    md = render_md(args.title, meta, a)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(a, f, indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()
