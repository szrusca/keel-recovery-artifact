import argparse
import json
import os
import time

from . import tasks
from .events import EventLog
from .providers import Pacer, call, estimate_tokens

T = tasks.get("t1")

STRATEGIES = ["same", "hot", "rephrase", "reset", "switch"]


def build_messages(system, steps_so_far, step, rephrased=False):
    msgs = [{"role": "system", "content": system}]
    for prev in steps_so_far:
        msgs.append({"role": "user", "content": T.user_message(prev)})
        msgs.append({"role": "assistant", "content": json.dumps(prev["after"], sort_keys=True)})
    if rephrased:
        ops = [o.strip() for o in step["instruction"].rstrip(".").split("; then ")]
        content = "Ledger before operations:\n" + json.dumps(step["before"], sort_keys=True) + "\nOperations, in order:\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(ops))
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": T.user_message(step)})
    return msgs


def strategy_request(strategy, cfg, history, step):
    base = (cfg["provider"], cfg["model"], cfg["reasoning"])
    if strategy == "same":
        return base, build_messages(T.SYSTEM_PRIMARY, history, step), cfg["temperature"]
    if strategy == "hot":
        return base, build_messages(T.SYSTEM_PRIMARY, history, step), 1.0
    if strategy == "rephrase":
        return base, build_messages(T.SYSTEM_REPHRASED, history, step, rephrased=True), cfg["temperature"]
    if strategy == "reset":
        return base, build_messages(T.SYSTEM_PRIMARY, [], step), cfg["temperature"]
    if strategy == "switch":
        return (cfg["switch_provider"], cfg["switch_model"], cfg["switch_reasoning"]), build_messages(T.SYSTEM_PRIMARY, history, step), cfg["temperature"]
    raise ValueError(strategy)


CALLS = {"n": 0}


def invoke(log, pacer, target, messages, temperature, cfg, **ctx):
    provider, model, reasoning = target
    key = f"{provider}:{model}"
    if cfg.get("max_calls") and CALLS["n"] >= cfg["max_calls"]:
        raise RuntimeError("max_calls reached")
    pacer.wait(key, provider, estimate_tokens(messages, cfg["expected_output"]))
    resp = call(provider, model, messages, temperature=temperature, max_tokens=cfg["max_tokens"], thinking_level=cfg["thinking_level"], reasoning=reasoning)
    CALLS["n"] += 1
    pacer.record(key, resp.input_tokens + resp.output_tokens + resp.thought_tokens or estimate_tokens(messages, cfg["expected_output"]))
    if resp.error_kind.startswith("infra"):
        pacer.backoff(key, resp, ctx.get("attempt", 1))
    return resp


def outcome_of(resp, expected):
    if resp.error_kind:
        return resp.error_kind
    kind, _ = T.verify(resp.text, expected)
    if kind != "ok" and resp.truncated:
        return "format_truncated"
    return kind


def run_trajectory(log, pacer, cfg, task):
    history = []
    summary = {"seed": task["seed"], "steps": []}
    log.emit("TaskStarted", seed=task["seed"], n_steps=len(task["steps"]), accounts=task["accounts"], config=cfg)
    for step in task["steps"]:
        target, messages, temperature = strategy_request("same", cfg, history, step)
        log.emit("StepRequested", seed=task["seed"], t=step["t"], path="main", context_turns=len(history))
        resp = invoke(log, pacer, target, messages, temperature, cfg, attempt=1)
        outcome = outcome_of(resp, step["after"])
        log.model_responded(resp, seed=task["seed"], t=step["t"], path="main", strategy="main", attempt=1, outcome=outcome, reasoning=target[2])
        step_record = {"t": step["t"], "main_outcome": outcome, "recoveries": {}}
        if outcome != "ok":
            for strategy in cfg["strategies"]:
                attempts = []
                recovered_at = None
                for attempt in range(1, cfg["k"] + 1):
                    tgt, msgs, temp = strategy_request(strategy, cfg, history, step)
                    log.emit("RetryScheduled", seed=task["seed"], t=step["t"], strategy=strategy, attempt=attempt, provider=tgt[0], model=tgt[1], reasoning=tgt[2], temperature=temp)
                    r = invoke(log, pacer, tgt, msgs, temp, cfg, attempt=attempt)
                    o = outcome_of(r, step["after"])
                    log.model_responded(r, seed=task["seed"], t=step["t"], path="retry", strategy=strategy, attempt=attempt, outcome=o, reasoning=tgt[2])
                    attempts.append(o)
                    if o == "ok":
                        recovered_at = attempt
                        break
                step_record["recoveries"][strategy] = {"attempts": attempts, "recovered_at": recovered_at}
        summary["steps"].append(step_record)
        history.append(step)
    log.emit("TaskEnded", seed=task["seed"], failures=sum(1 for s in summary["steps"] if s["main_outcome"] != "ok"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t1", choices=["t1", "t2"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--switch-provider", default="gemini")
    ap.add_argument("--switch-model", default="gemini-3.6-flash")
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--switch-reasoning", default=None)
    ap.add_argument("--n-traj", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--ops", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--rpm", default="groq=20,gemini=8,nvidia=20,cohere=10,openrouter=10,cerebras=14")
    ap.add_argument("--tpm", default="groq=7000")
    ap.add_argument("--expected-output", type=int, default=400)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--out", default="runs/pilot")
    args = ap.parse_args()

    global T
    T = tasks.get(args.task)
    cfg = {
        "task": args.task,
        "provider": args.provider,
        "model": args.model,
        "switch_provider": args.switch_provider,
        "switch_model": args.switch_model,
        "reasoning": args.reasoning,
        "switch_reasoning": args.switch_reasoning,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking_level": args.thinking_level,
        "k": args.k,
        "n_steps": args.n_steps,
        "ops_per_step": args.ops,
        "n_accounts": args.accounts,
        "strategies": args.strategies.split(","),
        "expected_output": args.expected_output,
        "max_calls": args.max_calls,
    }
    rpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.rpm.split(",")}
    tpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.tpm.split(",") if kv}
    pacer = Pacer(rpm, tpm)
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.provider}_{args.model.replace('/', '-')}{'_r-' + args.reasoning if args.reasoning else ''}_{stamp}"
    log = EventLog(os.path.join(args.out, f"{tag}.events.jsonl"))
    summaries = []
    for i in range(args.n_traj):
        task = T.generate(args.seed_start + i, n_accounts=args.accounts, n_steps=args.n_steps, ops_per_step=args.ops)
        try:
            s = run_trajectory(log, pacer, cfg, task)
        except RuntimeError as e:
            print("stopped:", e, flush=True)
            break
        summaries.append(s)
        fails = sum(1 for st in s["steps"] if st["main_outcome"] != "ok")
        print(f"traj {i + 1}/{args.n_traj} seed={task['seed']} main_failures={fails}/{args.n_steps} calls_so_far={CALLS['n']}", flush=True)
    with open(os.path.join(args.out, f"{tag}.summary.json"), "w") as f:
        json.dump({"config": cfg, "trajectories": summaries}, f, indent=1)
    print("events:", log.path)


if __name__ == "__main__":
    main()
