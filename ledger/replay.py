import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict

from . import frontier, tasks
from .events import EventLog, read_events
from .pricing import cost_usd
from .providers import Pacer, Response


def request_key(provider, model, reasoning, temperature, messages):
    payload = json.dumps([provider, model, reasoning or "", round(temperature, 3), messages], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


class Uncovered(Exception):
    pass


class NullLog:
    def emit(self, kind, **fields):
        return {"kind": kind, **fields}


class IndexingRunner:
    def __init__(self, sequences, index, cfg):
        self.sequences = sequences
        self.index = index
        self.cfg = cfg
        self.cursor = defaultdict(int)
        self.current = None
        self.log = NullLog()

    def ask(self, provider, model, messages, temperature, budget, reasoning=None, **ctx):
        seq = self.sequences[self.current]
        key = request_key(provider, model, reasoning, temperature, messages)
        e = None
        for _ in range(self.cfg.get("infra_retries", 2) + 1):
            i = self.cursor[self.current]
            if i >= len(seq):
                raise Uncovered("log exhausted")
            e = seq[i]
            self.cursor[self.current] = i + 1
            budget.charge({"usd": e["usd"], "input_tokens": e["input_tokens"], "output_tokens": e["output_tokens"], "thought_tokens": e["thought_tokens"]})
            if not (e.get("error_kind") or "").startswith("infra"):
                break
        if not (e.get("error_kind") or ""):
            self.index[key].append(e)
        return Response(provider, model, e["status"], e["text"], e["input_tokens"], e["output_tokens"], e["thought_tokens"], e["latency_ms"], e.get("error_kind") or "", "", {})


class ReplayRunner:
    def __init__(self, index, cfg, log, pick, live=None):
        self.index = index
        self.cfg = cfg
        self.log = log
        self.pick = pick
        self.live = live
        self.hits = 0
        self.misses = 0
        self.rng = random.Random(0)
        self.cursor = defaultdict(int)

    def reset_cursors(self):
        self.cursor = defaultdict(int)

    def ask(self, provider, model, messages, temperature, budget, reasoning=None, **ctx):
        key = request_key(provider, model, reasoning, temperature, messages)
        candidates = self.index.get(key)
        if candidates and self.pick == "sequential" and self.cursor[key] >= len(candidates):
            candidates = None
        if candidates:
            self.hits += 1
            if self.pick == "sequential":
                e = candidates[self.cursor[key]]
                self.cursor[key] += 1
            elif self.pick == "first":
                e = candidates[0]
            else:
                e = self.rng.choice(candidates)
            resp = Response(provider, model, 200, e["text"], e["input_tokens"], e["output_tokens"], e["thought_tokens"], e["latency_ms"], e.get("error_kind") or "", "", {})
            usd = cost_usd(provider, model, e["input_tokens"], e["output_tokens"], e["thought_tokens"])
            self.log.emit("ReplayHit", key=key, candidates=len(candidates), **ctx)
            budget.charge({"usd": usd, "input_tokens": e["input_tokens"], "output_tokens": e["output_tokens"], "thought_tokens": e["thought_tokens"]})
            return resp
        self.misses += 1
        if self.live is None:
            raise Uncovered(key)
        resp = self.live.ask(provider, model, messages, temperature, budget, reasoning=reasoning, **ctx)
        self.index[key].append({"text": resp.text, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "thought_tokens": resp.thought_tokens, "latency_ms": resp.latency_ms, "error_kind": resp.error_kind, "usd": cost_usd(provider, model, resp.input_tokens, resp.output_tokens, resp.thought_tokens)})
        return resp


def load_log(path):
    events = list(read_events(path))
    cfg = None
    sequences = defaultdict(list)
    rows = {}
    for e in events:
        if e["kind"] == "TaskStarted" and cfg is None:
            cfg = None
        if e["kind"] == "ModelResponded":
            sequences[(e["seed"], e["policy"])].append(e)
        if e["kind"] == "TaskEnded":
            rows[(e["seed"], e["policy"])] = e
    return sequences, rows


def build_index(sequences, cfg, policies, seeds):
    index = defaultdict(list)
    runner = IndexingRunner(sequences, index, cfg)
    for seed in seeds:
        task = tasks.get(cfg.get("task")).generate(seed, n_accounts=cfg["accounts"], n_steps=cfg["n_steps"], ops_per_step=cfg["ops"])
        for policy in policies:
            if (seed, policy) not in sequences:
                continue
            runner.current = (seed, policy)
            verifier = frontier.Verifier(cfg["verifier"], random.Random(seed), cfg.get("fn_rate", 0.0), cfg.get("fp_rate", 0.0))
            try:
                frontier.run_task(runner, cfg, task, policy, verifier, seed)
            except Uncovered:
                pass
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="rows.json of the source run (carries config)")
    ap.add_argument("--source-policies", default="none,retry,reset,rollback,escalate,budget")
    ap.add_argument("--target-policy", required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--usd-cap", type=float)
    ap.add_argument("--checkpoint-every", type=int)
    ap.add_argument("--max-rollbacks", type=int)
    ap.add_argument("--verifier")
    ap.add_argument("--pick", default="sequential", choices=["sequential", "first", "random"])
    ap.add_argument("--live", action="store_true", help="fall back to real calls on misses")
    ap.add_argument("--out", default="runs/replay")
    args = ap.parse_args()

    src = json.load(open(args.rows))
    cfg = dict(src["config"])
    events_path = args.rows.replace(".rows.json", ".events.jsonl")
    sequences, real_rows = load_log(events_path)
    seeds = sorted({s for s, _ in sequences})
    index = build_index(sequences, cfg, args.source_policies.split(","), seeds)

    tcfg = dict(cfg)
    for name in ["k", "usd_cap", "checkpoint_every", "max_rollbacks", "verifier"]:
        v = getattr(args, name)
        if v is not None:
            tcfg[name] = v
    os.makedirs(args.out, exist_ok=True)
    tag = f"replay_{os.path.basename(args.rows).replace('.rows.json', '')}_{args.target_policy}_{args.pick}_{time.strftime('%Y%m%d-%H%M%S')}"
    log = EventLog(os.path.join(args.out, tag + ".events.jsonl"))
    live = None
    if args.live:
        rpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in cfg["rpm"].split(",")}
        live = frontier.Runner(log, Pacer(rpm, {}), tcfg)
    runner = ReplayRunner(index, tcfg, log, args.pick, live)
    rows = []
    uncovered = 0
    for seed in seeds:
        task = tasks.get(cfg.get("task")).generate(seed, n_accounts=cfg["accounts"], n_steps=cfg["n_steps"], ops_per_step=cfg["ops"])
        verifier = frontier.Verifier(tcfg["verifier"], random.Random(seed), tcfg.get("fn_rate", 0.0), tcfg.get("fp_rate", 0.0))
        runner.reset_cursors()
        try:
            row = frontier.run_task(runner, tcfg, task, args.target_policy, verifier, seed)
            row["replayed"] = True
            rows.append(row)
        except Uncovered:
            uncovered += 1
            rows.append({"seed": seed, "policy": args.target_policy, "uncovered": True})
    covered = [r for r in rows if not r.get("uncovered")]
    summary = {
        "target_policy": args.target_policy,
        "target_config_overrides": {k: getattr(args, k) for k in ["k", "usd_cap", "checkpoint_every", "max_rollbacks", "verifier"] if getattr(args, k) is not None},
        "tasks": len(seeds),
        "tasks_fully_covered": len(covered),
        "request_hits": runner.hits,
        "request_misses": runner.misses,
        "replay_success_rate": (sum(1 for r in covered if r["success"]) / len(covered)) if covered else None,
        "replay_local_progress": (sum(r["local_progress"] for r in covered) / len(covered)) if covered else None,
        "replay_usd_per_task": (sum(r["usd"] for r in covered) / len(covered)) if covered else None,
    }
    real = [real_rows[(s, args.target_policy)] for s in seeds if (s, args.target_policy) in real_rows]
    if real:
        summary["real_success_rate"] = sum(1 for r in real if r["success"]) / len(real)
        summary["real_local_progress"] = sum(r["local_progress"] for r in real) / len(real)
        summary["real_usd_per_task"] = sum(r["usd"] for r in real) / len(real)
        agree = sum(1 for r in covered if real_rows.get((r["seed"], args.target_policy), {}).get("success") == r["success"])
        summary["per_task_success_agreement"] = agree / len(covered) if covered else None
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
