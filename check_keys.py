import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATHS = [os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")]


def load_env():
    for path in ENV_PATHS:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def request_json(url, body, headers, method="POST"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {}
        return e.code, dict(e.headers), payload
    except Exception as e:
        return None, {}, {"error": str(e)}


def rate_limit_headers(headers):
    return {k: v for k, v in headers.items() if "ratelimit" in k.lower() or "retry-after" in k.lower()}


PROMPT = "Reply with the single word: pong"


def openai_compatible(url, key, model, extra_headers=None):
    headers = {"Authorization": f"Bearer {key}"}
    headers.update(extra_headers or {})
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8}
    status, hdrs, payload = request_json(url, body, headers)
    try:
        text = payload["choices"][0]["message"]["content"].strip()
    except Exception:
        text = json.dumps(payload)[:200]
    return status, f"{model} -> {text}", rate_limit_headers(hdrs)


def check_gemini(key):
    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": PROMPT}]}]}
    status, hdrs, payload = request_json(url, body, {})
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        text = json.dumps(payload)[:200]
    return status, f"{model} -> {text}", rate_limit_headers(hdrs)


def check_cohere(key):
    model = "command-a-03-2025"
    url = "https://api.cohere.com/v2/chat"
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8}
    status, hdrs, payload = request_json(url, body, {"Authorization": f"Bearer {key}"})
    try:
        text = payload["message"]["content"][0]["text"].strip()
    except Exception:
        text = json.dumps(payload)[:200]
    return status, f"{model} -> {text}", rate_limit_headers(hdrs)


def check_openrouter(key):
    status, _, payload = request_json("https://openrouter.ai/api/v1/models", None, {"Authorization": f"Bearer {key}"}, method="GET")
    free = []
    for m in payload.get("data", []):
        pricing = m.get("pricing", {})
        if pricing.get("prompt") in ("0", 0, "0.0") and pricing.get("completion") in ("0", 0, "0.0"):
            free.append(m["id"])
    model = next((m for m in free if "llama-3.3-70b" in m), free[0] if free else "meta-llama/llama-3.3-70b-instruct:free")
    status, text, rl = openai_compatible("https://openrouter.ai/api/v1/chat/completions", key, model)
    return status, f"{text} | {len(free)} free models listed", rl


def check_cloudflare(token):
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"
    return openai_compatible(url, token, "@cf/openai/gpt-oss-120b")


def list_then_chat(models_url, chat_url, key, prefer):
    status, _, payload = request_json(models_url, None, {"Authorization": f"Bearer {key}"}, method="GET")
    ids = [m.get("id") or m.get("name") for m in (payload.get("data") or payload.get("models") or [])]
    model = next((i for i in ids if any(p in (i or "").lower() for p in prefer)), ids[0] if ids else prefer[0])
    status, text, rl = openai_compatible(chat_url, key, model)
    return status, f"{text} | {len(ids)} models listed", rl


CHECKS = [
    ("Google AI Studio", "GEMINI_API_KEY", check_gemini),
    ("Groq", "GROQ_API_KEY", lambda k: openai_compatible("https://api.groq.com/openai/v1/chat/completions", k, "llama-3.3-70b-versatile")),
    ("Cerebras", "CEREBRAS_API_KEY", lambda k: openai_compatible("https://api.cerebras.ai/v1/chat/completions", k, "llama3.1-8b")),
    ("Mistral", "MISTRAL_API_KEY", lambda k: openai_compatible("https://api.mistral.ai/v1/chat/completions", k, "mistral-small-latest")),
    ("Cohere", "COHERE_API_KEY", check_cohere),
    ("OpenRouter", "OPENROUTER_API_KEY", check_openrouter),
    ("NVIDIA NIM", "NVIDIA_API_KEY", lambda k: openai_compatible("https://integrate.api.nvidia.com/v1/chat/completions", k, "meta/llama-3.1-8b-instruct")),
    ("Hugging Face", "HF_TOKEN", lambda k: openai_compatible("https://router.huggingface.co/v1/chat/completions", k, "meta-llama/Llama-3.1-8B-Instruct")),
    ("Cloudflare Workers AI", "CLOUDFLARE_API_TOKEN", check_cloudflare),
    ("OVHcloud AI Endpoints", "OVH_API_KEY", lambda k: list_then_chat("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions", k, ["gpt-oss-120b", "qwen"])),
    ("Ollama Cloud", "OLLAMA_API_KEY", lambda k: list_then_chat("https://ollama.com/v1/models", "https://ollama.com/v1/chat/completions", k, ["deepseek-v4", "qwen3.5", "gpt-oss"])),
    ("Z AI", "ZAI_API_KEY", lambda k: openai_compatible("https://api.z.ai/api/paas/v4/chat/completions", k, "glm-4.7-flash")),
    ("Zhipu BigModel", "ZHIPU_API_KEY", lambda k: openai_compatible("https://open.bigmodel.cn/api/paas/v4/chat/completions", k, "glm-4.7-flash")),
    ("Kilo Code", "KILOCODE_API_KEY", lambda k: list_then_chat("https://api.kilo.ai/api/openrouter/models", "https://api.kilo.ai/api/openrouter/chat/completions", k, ["nemotron", "free"])),
    ("ModelScope", "MODELSCOPE_API_KEY", lambda k: openai_compatible("https://api-inference.modelscope.cn/v1/chat/completions", k, "Qwen/Qwen3.8-27B")),
    ("AMD Developer", "AMD_API_KEY", lambda k: openai_compatible("https://developer.amd.com.cn/radeon/api/v1/chat/completions", k, "DeepSeek-V4-Flash")),
    ("GitHub Models", "GITHUB_MODELS_TOKEN", lambda k: openai_compatible("https://models.github.ai/inference/chat/completions", k, "openai/gpt-4o-mini", {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})),
]


def main():
    load_env()
    failures = 0
    for name, var, fn in CHECKS:
        key = os.environ.get(var, "")
        if not key:
            print(f"[SKIP] {name}: {var} not set")
            continue
        status, text, rl = fn(key)
        ok = status == 200
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: HTTP {status} -> {text}")
        for k, v in sorted(rl.items()):
            print(f"       {k}: {v}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
