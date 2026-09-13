import argparse
import json
import os
import time
from collections import Counter

from . import tasks_t1
from .events import EventLog
from .providers import Pacer, call, estimate_tokens


def sample_steps(n, seed_start, n_steps, ops, accounts):
    steps = []
    seed = seed_start
    while len(steps) < n:
        task = tasks_t1.generate(seed, n_accounts=accounts, n_steps=n_steps, ops_per_step=ops)
        for t in range(0, n_steps, max(1, n_steps // 4)):
            if len(steps) >= n:
                break
            history = task["steps"][:t]
            step = task["steps"][t]
            steps.append((seed, step, history))
        seed += 1
    return steps


def parse_timeouts(spec):
    default = 240.0
    per = {}
    for kv in spec.split(","):
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=")
            per[k] = float(v)
        else:
            default = float(kv)
    return default, per


def build_messages(history, step):
    msgs = [{"role": "system", "content": tasks_t1.SYSTEM_PRIMARY}]
    for prev in history:
        msgs.append({"role": "user", "content": tasks_t1.user_message(prev)})
        msgs.append({"role": "assistant", "content": json.dumps(prev["after"], sort_keys=True)})
    msgs.append({"role": "user", "content": tasks_t1.user_message(step)})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", required=True, help="comma list of provider|model[|reasoning]")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--ops", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--rpm", default="groq=10,ovh=2,cloudflare=6,nvidia=6,kilocode=3,openrouter=4,ollama=3,modelscope=6")
    ap.add_argument("--tpm", default="groq=6000")
    ap.add_argument("--limit", default="", help="per-stack call caps, e.g. cloudflare=25")
    ap.add_argument("--timeout", default="240", help="seconds, optionally per provider, e.g. 240,nvidia=60")
    ap.add_argument("--out", default="runs/stacks")
    args = ap.parse_args()

    stacks = []
    for spec in args.stacks.split(","):
        parts = spec.split("|")
        stacks.append((parts[0], parts[1], parts[2] if len(parts) > 2 and parts[2] else None, parts[3] if len(parts) > 3 else ""))
    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.limit.split(",") if kv}
    default_timeout, timeouts = parse_timeouts(args.timeout)
    rpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.rpm.split(",")}
    tpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.tpm.split(",") if kv}
    pacer = Pacer(rpm, tpm)
    os.makedirs(args.out, exist_ok=True)
    tag = f"stacks_{args.label}_{time.strftime('%Y%m%d-%H%M%S')}"
    log = EventLog(os.path.join(args.out, tag + ".events.jsonl"))
    steps = sample_steps(args.n, args.seed_start, args.n_steps, args.ops, args.accounts)
    results = []
    used = Counter()
    for idx, (seed, step, history) in enumerate(steps):
        msgs = build_messages(history, step)
        row = {"idx": idx, "seed": seed, "t": step["t"], "outputs": {}}
        for provider, model, reasoning, label in stacks:
            if provider in caps and used[provider] >= caps[provider]:
                continue
            key = f"{provider}:{model}" + (f"[{reasoning}]" if reasoning else "") + (f"#{label}" if label else "")
            pacer.wait(key, provider, estimate_tokens(msgs, 600))
            timeout = timeouts.get(provider, default_timeout)
            resp = call(provider, model, msgs, temperature=args.temperature, max_tokens=args.max_tokens, timeout=timeout, reasoning=reasoning)
            used[provider] += 1
            pacer.record(key, resp.input_tokens + resp.output_tokens + resp.thought_tokens or 1000)
            if resp.error_kind.startswith("infra"):
                pacer.backoff(key, resp, 1)
                resp = call(provider, model, msgs, temperature=args.temperature, max_tokens=args.max_tokens, timeout=timeout, reasoning=reasoning)
                used[provider] += 1
            log.model_responded(resp, idx=idx, seed=seed, t=step["t"], stack=key)
            kind, obj = tasks_t1.verify(resp.text, step["after"])
            row["outputs"][key] = {"status": resp.status, "error": resp.error_kind, "kind": kind, "ledger": obj, "text": resp.text.strip(), "latency_ms": resp.latency_ms, "tokens": resp.input_tokens + resp.output_tokens + resp.thought_tokens}
        results.append(row)
        print(f"step {idx + 1}/{len(steps)} seed={seed} t={step['t']} " + " ".join(f"{k.split(':')[0]}={v['kind']}" for k, v in row["outputs"].items()), flush=True)
    json.dump({"stacks": [f"{p}:{m}" + (f"[{r}]" if r else "") + (f"#{l}" if l else "") for p, m, r, l in stacks], "results": results}, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
    print("saved", os.path.join(args.out, tag + ".json"))


if __name__ == "__main__":
    main()
