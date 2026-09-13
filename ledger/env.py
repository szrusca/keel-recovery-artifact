import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_PATHS = [os.path.join(ROOT, ".env"), os.path.join(ROOT, "..", ".env")]


def load_env():
    set_from_file = set()
    for path in ENV_PATHS:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not v:
                continue
            if k in set_from_file or not os.environ.get(k):
                os.environ[k] = v
                set_from_file.add(k)
