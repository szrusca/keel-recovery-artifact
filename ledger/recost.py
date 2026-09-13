import argparse
import json
from collections import defaultdict

from .events import read_events
from .pricing import cost_usd


def recost(events_path, rows_path):
    usd = defaultdict(float)
    for e in read_events(events_path):
        if e["kind"] != "ModelResponded":
            continue
        usd[(e["seed"], e["policy"])] += cost_usd(e["provider"], e["model"], e["input_tokens"], e["output_tokens"], e["thought_tokens"])
    d = json.load(open(rows_path))
    changed = 0
    for r in d["rows"]:
        new = usd.get((r["seed"], r["policy"]), r["usd"])
        if abs(new - r["usd"]) > 1e-9:
            r["usd"] = new
            changed += 1
    json.dump(d, open(rows_path, "w"), indent=1)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("rows")
    a = ap.parse_args()
    print("rows recosted:", recost(a.events, a.rows))


if __name__ == "__main__":
    main()
