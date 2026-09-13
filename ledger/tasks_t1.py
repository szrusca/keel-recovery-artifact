import json
import random
import re

ACCOUNTS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]

SYSTEM_PRIMARY = (
    "You maintain a ledger of integer account balances. Each turn you receive the current ledger "
    "and one instruction containing one or more operations to apply in order. Apply them exactly "
    "and reply with ONLY the complete updated ledger as a single JSON object mapping every account "
    "name to its integer balance. No explanation, no code fences, no extra keys."
)

SYSTEM_REPHRASED = (
    "Act as a deterministic bookkeeping engine. Input: a JSON ledger of accounts with whole-number "
    "balances plus a list of operations. Output: the ledger after all operations have been applied "
    "sequentially, as one JSON object with exactly the same account names and integer values. "
    "Output the JSON object and nothing else."
)


def _gen_ops(rng, accounts, n_ops):
    ops = []
    for _ in range(n_ops):
        kind = rng.choice(["deposit", "withdraw", "transfer", "set", "double", "halve", "conditional", "swap", "percent"])
        a, b = rng.sample(accounts, 2)
        n = rng.randint(1, 250)
        if kind == "deposit":
            ops.append(("deposit", a, n))
        elif kind == "withdraw":
            ops.append(("withdraw", a, n))
        elif kind == "transfer":
            ops.append(("transfer", a, b, n))
        elif kind == "set":
            ops.append(("set", a, n))
        elif kind == "double":
            ops.append(("double", a))
        elif kind == "halve":
            ops.append(("halve", a))
        elif kind == "conditional":
            ops.append(("conditional", a, b, n))
        elif kind == "swap":
            ops.append(("swap", a, b))
        else:
            ops.append(("percent", a, rng.choice([10, 20, 25, 50])))
    return ops


def _apply(state, op):
    s = dict(state)
    kind = op[0]
    if kind == "deposit":
        s[op[1]] += op[2]
    elif kind == "withdraw":
        s[op[1]] -= op[2]
    elif kind == "transfer":
        s[op[1]] -= op[3]
        s[op[2]] += op[3]
    elif kind == "set":
        s[op[1]] = op[2]
    elif kind == "double":
        s[op[1]] *= 2
    elif kind == "halve":
        s[op[1]] = s[op[1]] // 2
    elif kind == "conditional":
        if s[op[1]] > s[op[2]]:
            s[op[1]] -= op[3]
        else:
            s[op[2]] -= op[3]
    elif kind == "swap":
        s[op[1]], s[op[2]] = s[op[2]], s[op[1]]
    elif kind == "percent":
        s[op[1]] = s[op[1]] + (s[op[1]] * op[2]) // 100
    return s


def _describe(op):
    kind = op[0]
    if kind == "deposit":
        return f"deposit {op[2]} into {op[1]}"
    if kind == "withdraw":
        return f"withdraw {op[2]} from {op[1]} (balances may go negative)"
    if kind == "transfer":
        return f"transfer {op[3]} from {op[1]} to {op[2]}"
    if kind == "set":
        return f"set {op[1]} to exactly {op[2]}"
    if kind == "double":
        return f"double the balance of {op[1]}"
    if kind == "halve":
        return f"halve the balance of {op[1]}, rounding toward negative infinity (floor division by 2)"
    if kind == "conditional":
        return f"if {op[1]} is strictly greater than {op[2]}, withdraw {op[3]} from {op[1]}; otherwise withdraw {op[3]} from {op[2]}"
    if kind == "swap":
        return f"swap the balances of {op[1]} and {op[2]}"
    if kind == "percent":
        return f"increase {op[1]} by {op[2]} percent of its current balance, discarding any fractional part of the increase (floor)"
    return str(op)


def generate(seed, n_accounts=6, n_steps=16, ops_per_step=3):
    rng = random.Random(seed)
    accounts = ACCOUNTS[:n_accounts]
    state = {a: rng.randint(50, 900) for a in accounts}
    steps = []
    for t in range(n_steps):
        ops = _gen_ops(rng, accounts, ops_per_step)
        new_state = state
        for op in ops:
            new_state = _apply(new_state, op)
        instruction = "; then ".join(_describe(op) for op in ops) + "."
        steps.append({"t": t + 1, "before": state, "instruction": instruction, "after": new_state, "ops": [list(op) for op in ops]})
        state = new_state
    return {"seed": seed, "accounts": accounts, "initial": steps[0]["before"], "steps": steps}


def apply_ops(state, ops):
    s = dict(state)
    for op in ops:
        s = _apply(s, tuple(op))
    return s


def user_message(step):
    return f"Current ledger: {json.dumps(step['before'], sort_keys=True)}\nInstruction: {step['instruction']}"


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def canonical_state(obj, expected_keys):
    if not isinstance(obj, dict) or set(obj) != set(expected_keys):
        return None
    try:
        return {k: int(v) for k, v in obj.items()}
    except (TypeError, ValueError):
        return None


def verify(text, expected):
    if text is None or not text.strip():
        return "format_empty", None
    m = _JSON_OBJ.search(text)
    if not m:
        return "format_nojson", None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return "format_badjson", None
    if not isinstance(obj, dict) or set(obj) != set(expected):
        return "format_keys", obj
    for k, v in obj.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            return "format_type", obj
    if all(int(obj[k]) == expected[k] for k in expected):
        return "ok", obj
    return "semantic", obj
