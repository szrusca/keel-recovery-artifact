import json
import random
import re

SKUS = ["bolt", "nut", "washer", "gear", "spring", "bracket"]

SYSTEM_PRIMARY = (
    "You operate a small warehouse order system. State is a JSON object with two keys: "
    "\"inventory\" mapping SKU to integer on-hand count, and \"orders\" mapping order id to an object "
    "{\"status\": one of NEW, RESERVED, SHIPPED, CANCELLED, \"items\": {SKU: integer quantity}}. "
    "Each turn you receive the current state and one instruction with one or more operations to apply in order. "
    "Rules: RESERVE decrements inventory by the order's items and sets status RESERVED (only from NEW; if any SKU "
    "would go below zero, do nothing for that operation). SHIP sets status SHIPPED (only from RESERVED). CANCEL applies only to NEW or "
    "RESERVED orders: it sets status CANCELLED and, if the order was RESERVED, returns its items to inventory; "
    "SHIPPED and CANCELLED orders cannot be cancelled again. RESTOCK adds to inventory. "
    "CREATE adds a NEW order with the given items. Operations that violate a rule are skipped without changing state. "
    "Reply with ONLY the complete updated state as a single JSON object. No explanation, no code fences."
)

SYSTEM_REPHRASED = (
    "Act as a deterministic warehouse state machine. Input: a JSON state {inventory, orders} and a numbered list of "
    "operations. Apply them strictly in order following these rules: RESERVE (NEW->RESERVED, decrement inventory, skip "
    "if any count would go negative), SHIP (RESERVED->SHIPPED), CANCEL (NEW or RESERVED -> CANCELLED, restore items only if it was "
    "RESERVED; SHIPPED or CANCELLED orders are not affected), RESTOCK (add to inventory), CREATE (new NEW order). Skip any operation that violates its precondition. "
    "Output the resulting JSON state and nothing else."
)


def _new_order(rng, skus):
    n = rng.randint(1, 3)
    chosen = rng.sample(skus, n)
    return {"status": "NEW", "items": {s: rng.randint(1, 6) for s in chosen}}


def _apply(state, op):
    s = {"inventory": dict(state["inventory"]), "orders": {k: {"status": v["status"], "items": dict(v["items"])} for k, v in state["orders"].items()}}
    kind = op[0]
    if kind == "CREATE":
        _, oid, items = op
        s["orders"][oid] = {"status": "NEW", "items": dict(items)}
    elif kind == "RESERVE":
        o = s["orders"].get(op[1])
        if o and o["status"] == "NEW" and all(s["inventory"].get(k, 0) >= q for k, q in o["items"].items()):
            for k, q in o["items"].items():
                s["inventory"][k] -= q
            o["status"] = "RESERVED"
    elif kind == "SHIP":
        o = s["orders"].get(op[1])
        if o and o["status"] == "RESERVED":
            o["status"] = "SHIPPED"
    elif kind == "CANCEL":
        o = s["orders"].get(op[1])
        if o and o["status"] in ("NEW", "RESERVED"):
            if o["status"] == "RESERVED":
                for k, q in o["items"].items():
                    s["inventory"][k] = s["inventory"].get(k, 0) + q
            o["status"] = "CANCELLED"
    elif kind == "RESTOCK":
        s["inventory"][op[1]] = s["inventory"].get(op[1], 0) + op[2]
    return s


def _describe(op):
    kind = op[0]
    if kind == "CREATE":
        return f"CREATE order {op[1]} with items {json.dumps(op[2], sort_keys=True)}"
    if kind == "RESTOCK":
        return f"RESTOCK {op[2]} units of {op[1]}"
    return f"{kind} order {op[1]}"


def _gen_ops(rng, state, n_ops, counter):
    ops = []
    cur = state
    for _ in range(n_ops):
        orders = list(cur["orders"].keys())
        choices = ["CREATE", "RESTOCK"] + (["RESERVE", "SHIP", "CANCEL"] * 2 if orders else [])
        kind = rng.choice(choices)
        if kind == "CREATE":
            counter[0] += 1
            op = ("CREATE", f"o{counter[0]}", _new_order(rng, SKUS)["items"])
        elif kind == "RESTOCK":
            op = ("RESTOCK", rng.choice(SKUS), rng.randint(2, 10))
        else:
            op = (kind, rng.choice(orders))
        ops.append(op)
        cur = _apply(cur, op)
    return ops


def generate(seed, n_accounts=6, n_steps=16, ops_per_step=3):
    rng = random.Random(seed)
    skus = SKUS[:n_accounts]
    inventory = {s: rng.randint(4, 15) for s in skus}
    orders = {}
    counter = [0]
    for _ in range(2):
        counter[0] += 1
        orders[f"o{counter[0]}"] = _new_order(rng, skus)
    state = {"inventory": inventory, "orders": orders}
    steps = []
    for t in range(n_steps):
        ops = _gen_ops(rng, state, ops_per_step, counter)
        new_state = state
        for op in ops:
            new_state = _apply(new_state, op)
        instruction = "; then ".join(_describe(op) for op in ops) + "."
        steps.append({"t": t + 1, "before": state, "instruction": instruction, "after": new_state, "ops": [list(op) for op in ops]})
        state = new_state
    return {"seed": seed, "accounts": skus, "initial": steps[0]["before"], "steps": steps}


def apply_ops(state, ops):
    s = state
    for op in ops:
        s = _apply(s, tuple(op) if op[0] != "CREATE" else ("CREATE", op[1], op[2]))
    return s


def user_message(step):
    return f"Current state: {json.dumps(step['before'], sort_keys=True)}\nInstruction: {step['instruction']}"


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _canon(obj):
    if not isinstance(obj, dict) or set(obj) != {"inventory", "orders"}:
        return None
    inv = obj["inventory"]
    orders = obj["orders"]
    if not isinstance(inv, dict) or not isinstance(orders, dict):
        return None
    out = {"inventory": {}, "orders": {}}
    for k, v in inv.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            return None
        out["inventory"][k] = int(v)
    for oid, o in orders.items():
        if not isinstance(o, dict) or "status" not in o or "items" not in o or not isinstance(o["items"], dict):
            return None
        items = {}
        for k, v in o["items"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
                return None
            items[k] = int(v)
        out["orders"][oid] = {"status": str(o["status"]), "items": items}
    return out


def canonical_state(obj, expected_keys):
    return _canon(obj)


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
    canon = _canon(obj)
    if canon is None:
        return "format_keys", obj
    if canon == expected:
        return "ok", canon
    return "semantic", canon
