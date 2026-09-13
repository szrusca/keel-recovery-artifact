import argparse
import json
import os
import random
import time

from . import tasks
from .events import EventLog
from .providers import Pacer, call, estimate_tokens

POLICIES = ["none", "retry", "reset", "rollback", "escalate", "budget", "escalate_step"]


class Budget:
    def __init__(self, usd_cap):
        self.usd_cap = usd_cap
        self.usd = 0.0
        self.tokens = 0
        self.calls = 0

    def charge(self, event):
        self.usd += event["usd"]
        self.tokens += event["input_tokens"] + event["output_tokens"] + event["thought_tokens"]
        self.calls += 1

    def exhausted(self):
        return self.usd_cap > 0 and self.usd >= self.usd_cap


T = tasks.get("t1")


class Verifier:
    def __init__(self, mode, rng, fn_rate=0.0, fp_rate=0.0):
        self.mode = mode
        self.rng = rng
        self.fn_rate = fn_rate
        self.fp_rate = fp_rate

    def check(self, text, expected):
        kind, obj = T.verify(text, expected)
        if self.mode == "format":
            return ("ok" if kind in ("ok", "semantic") else kind), obj
        if self.mode == "oracle":
            return kind, obj
        if self.mode == "noisy":
            if kind == "ok" and self.rng.random() < self.fp_rate:
                return "semantic", obj
            if kind == "semantic" and self.rng.random() < self.fn_rate:
                return "ok", obj
            return kind, obj
        raise ValueError(self.mode)


class Runner:
    def __init__(self, log, pacer, cfg):
        self.log = log
        self.pacer = pacer
        self.cfg = cfg

    def ask(self, provider, model, messages, temperature, budget, reasoning=None, **ctx):
        key = f"{provider}:{model}"
        for infra_attempt in range(1, self.cfg["infra_retries"] + 2):
            if budget.calls >= self.cfg["max_calls_per_task"]:
                raise RuntimeError("max_calls_per_task")
            self.pacer.wait(key, provider, estimate_tokens(messages, self.cfg["expected_output"]))
            resp = call(provider, model, messages, temperature=temperature, max_tokens=self.cfg["max_tokens"], thinking_level=self.cfg["thinking_level"], reasoning=reasoning)
            self.pacer.record(key, resp.input_tokens + resp.output_tokens + resp.thought_tokens or estimate_tokens(messages, self.cfg["expected_output"]))
            ev = self.log.model_responded(resp, infra_attempt=infra_attempt, **ctx)
            budget.charge(ev)
            if not resp.error_kind.startswith("infra"):
                return resp
            self.pacer.backoff(key, resp, infra_attempt)
        return resp


def messages_from(history, state, step, system=None):
    msgs = [{"role": "system", "content": system or T.SYSTEM_PRIMARY}]
    for h in history:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": f"Current ledger: {json.dumps(state, sort_keys=True)}\nInstruction: {step['instruction']}"})
    return msgs


def run_task(runner, cfg, task, policy, verifier, seed):
    global T
    T = tasks.get(cfg.get("task"))
    log = runner.log
    budget = Budget(cfg["usd_cap"] if policy == "budget" else 0.0)
    history = []
    state = dict(task["initial"])
    checkpoint = {"t": 0, "state": dict(state), "history": []}
    steps = task["steps"]
    t = 0
    provider, model, reasoning = cfg["provider"], cfg["model"], cfg["reasoning"]
    base_target = (provider, model, reasoning)
    escalated = False
    step_escalated = False
    escalations = 0
    aborted = False
    per_step_correct = []
    per_step_local = []
    rollbacks = 0
    retries = 0
    runaway = False
    log.emit("TaskStarted", seed=task["seed"], policy=policy, verifier=verifier.mode, n_steps=len(steps))
    while t < len(steps):
        step = steps[t]
        truth_state = T.apply_ops(state, step["ops"])
        msgs = messages_from(history, state, step)
        attempt = 0
        outcome = None
        text = ""
        obj = None
        while True:
            attempt += 1
            if budget.exhausted():
                aborted = True
                break
            try:
                resp = runner.ask(provider, model, msgs, cfg["temperature"], budget, reasoning=reasoning, seed=task["seed"], policy=policy, t=step["t"], attempt=attempt, path="main" if attempt == 1 else "retry")
            except RuntimeError:
                runaway = True
                aborted = True
                break
            text = resp.text
            if resp.error_kind:
                outcome, obj = resp.error_kind, None
            else:
                outcome, obj = verifier.check(text, truth_state)
                if outcome != "ok" and resp.truncated:
                    outcome = "format_truncated"
            if outcome == "ok" or policy == "none":
                break
            if budget.exhausted():
                aborted = True
                break
            if attempt > cfg["k"]:
                if policy in ("escalate", "escalate_step") and not (escalated if policy == "escalate" else step_escalated):
                    escalated = True
                    step_escalated = True
                    escalations += 1
                    provider, model, reasoning = cfg["switch_provider"], cfg["switch_model"], cfg["switch_reasoning"]
                    attempt = 0
                    log.emit("Escalated", seed=task["seed"], policy=policy, t=step["t"], to=f"{provider}:{model}")
                    continue
                break
            retries += 1
            log.emit("RetryScheduled", seed=task["seed"], policy=policy, t=step["t"], attempt=attempt + 1, reason=outcome)
            if policy == "reset":
                msgs = messages_from([], state, step)
            elif policy == "rollback":
                if rollbacks >= cfg["max_rollbacks"]:
                    break
                rollbacks += 1
                log.emit("RolledBack", seed=task["seed"], policy=policy, from_t=step["t"], to_t=checkpoint["t"])
                state = dict(checkpoint["state"])
                history = list(checkpoint["history"])
                t = checkpoint["t"]
                step = steps[t]
                truth_state = T.apply_ops(state, step["ops"])
                msgs = messages_from(history, state, step)
        if aborted:
            log.emit("Runaway" if runaway else "BudgetExhausted", seed=task["seed"], policy=policy, t=step["t"], usd=budget.usd, calls=budget.calls)
            break
        if policy == "escalate_step" and step_escalated:
            provider, model, reasoning = base_target
            step_escalated = False
            log.emit("ReturnedToBase", seed=task["seed"], policy=policy, t=step["t"])
        candidate = T.canonical_state(obj, state.keys()) if obj is not None else None
        new_state = candidate if candidate is not None else state
        correct = new_state == step["after"]
        local_correct = new_state == truth_state
        if len(per_step_correct) > t:
            per_step_correct = per_step_correct[:t]
            per_step_local = per_step_local[:t]
        per_step_correct.append(correct)
        per_step_local.append(local_correct)
        log.emit("StepCommitted", seed=task["seed"], policy=policy, t=step["t"], outcome=outcome, global_correct=correct, local_correct=local_correct, state=new_state)
        history.append({"user": msgs[-1]["content"], "assistant": json.dumps(new_state, sort_keys=True)})
        state = new_state
        if outcome == "ok" and (t + 1) % cfg["checkpoint_every"] == 0:
            checkpoint = {"t": t + 1, "state": dict(state), "history": list(history)}
            log.emit("CheckpointTaken", seed=task["seed"], policy=policy, t=t + 1)
        t += 1
    final_ok = (not aborted) and state == steps[-1]["after"]
    progress = sum(per_step_correct) / len(steps)
    local_progress = sum(per_step_local) / len(steps)
    log.emit("TaskEnded", seed=task["seed"], policy=policy, success=final_ok, progress=progress, local_progress=local_progress, usd=budget.usd, tokens=budget.tokens, calls=budget.calls, retries=retries, rollbacks=rollbacks, escalated=escalated, escalations=escalations, aborted=aborted)
    return {"seed": task["seed"], "policy": policy, "success": final_ok, "progress": progress, "local_progress": local_progress, "usd": budget.usd, "tokens": budget.tokens, "calls": budget.calls, "retries": retries, "rollbacks": rollbacks, "escalated": escalated, "escalations": escalations, "aborted": aborted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t1", choices=["t1", "t2"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--switch-provider", default="nvidia")
    ap.add_argument("--switch-model", default="nvidia/nemotron-3-super-120b-a12b")
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--switch-reasoning", default=None)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--verifier", default="oracle", choices=["format", "oracle", "noisy"])
    ap.add_argument("--fn-rate", type=float, default=0.2)
    ap.add_argument("--fp-rate", type=float, default=0.05)
    ap.add_argument("--verifier-seed", type=int, default=0)
    ap.add_argument("--n-traj", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--ops", type=int, default=5)
    ap.add_argument("--accounts", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=4)
    ap.add_argument("--max-rollbacks", type=int, default=4)
    ap.add_argument("--infra-retries", type=int, default=2)
    ap.add_argument("--max-calls-per-task", type=int, default=0)
    ap.add_argument("--usd-cap", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--expected-output", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=2000)
    ap.add_argument("--rpm", default="groq=20,gemini=8,nvidia=20,cohere=10,openrouter=10")
    ap.add_argument("--tpm", default="groq=7000")
    ap.add_argument("--out", default="runs/frontier")
    args = ap.parse_args()
    cfg = vars(args).copy()
    cfg["policies"] = args.policies.split(",")
    global T
    T = tasks.get(args.task)
    if not cfg["max_calls_per_task"]:
        cfg["max_calls_per_task"] = 6 * args.n_steps
    rpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.rpm.split(",")}
    tpm = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.tpm.split(",") if kv}
    pacer = Pacer(rpm, tpm)
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.provider}_{args.model.replace('/', '-')}_{args.verifier}_{stamp}"
    log = EventLog(os.path.join(args.out, f"{tag}.events.jsonl"))
    runner = Runner(log, pacer, cfg)
    rows = []
    for i in range(args.n_traj):
        task = T.generate(args.seed_start + i, n_accounts=args.accounts, n_steps=args.n_steps, ops_per_step=args.ops)
        for policy in cfg["policies"]:
            verifier = Verifier(args.verifier, random.Random(args.seed_start + i + 1000003 * args.verifier_seed), args.fn_rate, args.fp_rate)
            row = run_task(runner, cfg, task, policy, verifier, args.seed_start + i)
            rows.append(row)
            print(f"seed={row['seed']} policy={policy:9s} success={row['success']} progress={row['progress']:.2f} local={row['local_progress']:.2f} usd={row['usd']:.5f} calls={row['calls']} retries={row['retries']} rollbacks={row['rollbacks']} escalated={row['escalated']} aborted={row['aborted']}", flush=True)
    with open(os.path.join(args.out, f"{tag}.rows.json"), "w") as f:
        json.dump({"config": cfg, "rows": rows}, f, indent=1)
    print("events:", log.path)


if __name__ == "__main__":
    main()
