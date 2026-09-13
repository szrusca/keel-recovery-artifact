import json
import os
import time
import uuid

from .pricing import PRICE_DATE, cost_usd


class EventLog:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        self.seq = 0

    def emit(self, kind, **fields):
        self.seq += 1
        record = {"run_id": self.run_id, "seq": self.seq, "ts": time.time(), "kind": kind}
        record.update(fields)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def model_responded(self, resp, **ctx):
        return self.emit(
            "ModelResponded",
            provider=resp.provider,
            model=resp.model,
            status=resp.status,
            error_kind=resp.error_kind,
            error_detail=resp.error_detail,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            thought_tokens=resp.thought_tokens,
            latency_ms=resp.latency_ms,
            usd=cost_usd(resp.provider, resp.model, resp.input_tokens, resp.output_tokens, resp.thought_tokens),
            price_date=PRICE_DATE,
            rate_limit=resp.rate_limit,
            text=resp.text,
            **ctx,
        )


def read_events(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
