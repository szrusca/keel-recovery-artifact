import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .env import load_env

load_env()

OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
    "ovh": ("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions", "OVH_API_KEY"),
    "ollama": ("https://ollama.com/v1/chat/completions", "OLLAMA_API_KEY"),
    "zai": ("https://api.z.ai/api/paas/v4/chat/completions", "ZAI_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4/chat/completions", "ZHIPU_API_KEY"),
    "kilocode": ("https://api.kilo.ai/api/openrouter/chat/completions", "KILOCODE_API_KEY"),
    "modelscope": ("https://api-inference.modelscope.cn/v1/chat/completions", "MODELSCOPE_API_KEY"),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1/chat/completions", "CLOUDFLARE_API_TOKEN"),
    "amd": ("https://developer.amd.com.cn/radeon/api/v1/chat/completions", "AMD_API_KEY"),
}

INFRA_STATUSES = {404, 408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


@dataclass
class Response:
    provider: str
    model: str
    status: int
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_ms: int = 0
    error_kind: str = ""
    error_detail: str = ""
    rate_limit: dict = field(default_factory=dict)
    truncated: bool = False

    @property
    def ok(self):
        return self.status == 200 and not self.error_kind


def _http(url, body, headers, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "ledger-agent-reliability/0.1 (+https://github.com; research)")
    req.add_header("Accept", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw, int((time.time() - t0) * 1000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(), int((time.time() - t0) * 1000)
    except Exception as e:
        return 0, {}, str(e).encode(), int((time.time() - t0) * 1000)


def _rate_limit(headers):
    return {k.lower(): v for k, v in headers.items() if "ratelimit" in k.lower() or k.lower() in ("retry-after", "cf-ai-neurons")}


def _classify(status, raw):
    if status == 200:
        return ""
    if status == 0:
        return "infra_network"
    if status in INFRA_STATUSES:
        return "infra_http_%d" % status
    return "client_http_%d" % status


def _openai_call(provider, model, messages, temperature, max_tokens, timeout, seed, reasoning, extra=None):
    url, key_var = OPENAI_COMPATIBLE[provider]
    url = url.format(account=os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))
    key = os.environ.get(key_var, "")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if reasoning == "nothink_prompt":
        messages = [dict(m) for m in messages]
        for m in messages:
            if m["role"] == "system":
                m["content"] = "/no_think\n" + m["content"]
                break
        body["messages"] = messages
    elif reasoning == "nothink":
        body["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
    elif reasoning == "think":
        body["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    elif reasoning:
        body["reasoning_effort"] = reasoning
    if extra:
        body.update(extra)
    if seed is not None and provider in ("groq", "cerebras"):
        body["seed"] = seed
    auth = {"Authorization": f"Bearer {key}"} if key else {}
    status, headers, raw, latency = _http(url, body, auth, timeout)
    if provider == "ovh" and status == 401 and key:
        status, headers, raw, latency = _http(url, body, {}, timeout)
    resp = Response(provider, model, status, "", latency_ms=latency, rate_limit=_rate_limit(headers))
    resp.error_kind = _classify(status, raw)
    if resp.error_kind:
        resp.error_detail = raw.decode(errors="replace")[:300]
        return resp
    try:
        payload = json.loads(raw)
        wrapped = payload.get("error") if isinstance(payload, dict) else None
        if wrapped and "choices" not in payload:
            code = int(wrapped.get("code") or 0) if str(wrapped.get("code", "")).isdigit() else 0
            resp.error_kind = ("infra_http_%d" % code) if code in INFRA_STATUSES else "infra_wrapped_error"
            resp.error_detail = raw.decode(errors="replace")[:300]
            return resp
        choice = payload["choices"][0]
        msg = choice["message"]
        resp.text = msg.get("content") or ""
        usage = payload.get("usage") or {}
        resp.input_tokens = int(usage.get("prompt_tokens") or 0)
        resp.output_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        resp.thought_tokens = int(details.get("reasoning_tokens") or 0)
        resp.truncated = choice.get("finish_reason") == "length"
        if resp.truncated and not resp.text.strip():
            resp.error_kind = "format_truncated"
    except Exception as e:
        resp.error_kind = "client_parse"
        resp.error_detail = f"{e}: {raw[:200]!r}"
    return resp


def _gemini_call(model, messages, temperature, max_tokens, timeout, thinking_level):
    key = os.environ.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    system = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingLevel": thinking_level},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
    status, headers, raw, latency = _http(url, body, {}, timeout)
    resp = Response("gemini", model, status, "", latency_ms=latency, rate_limit=_rate_limit(headers))
    resp.error_kind = _classify(status, raw)
    if resp.error_kind:
        resp.error_detail = raw.decode(errors="replace")[:300]
        return resp
    try:
        payload = json.loads(raw)
        cand = payload["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        resp.text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = payload.get("usageMetadata") or {}
        resp.input_tokens = int(usage.get("promptTokenCount") or 0)
        resp.output_tokens = int(usage.get("candidatesTokenCount") or 0)
        resp.thought_tokens = int(usage.get("thoughtsTokenCount") or 0)
        resp.truncated = cand.get("finishReason") == "MAX_TOKENS"
        if resp.truncated and not resp.text.strip():
            resp.error_kind = "format_truncated"
    except Exception as e:
        resp.error_kind = "client_parse"
        resp.error_detail = f"{e}: {raw[:200]!r}"
    return resp


def _cohere_call(model, messages, temperature, max_tokens, timeout):
    key = os.environ.get("COHERE_API_KEY", "")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    status, headers, raw, latency = _http("https://api.cohere.com/v2/chat", body, {"Authorization": f"Bearer {key}"}, timeout)
    resp = Response("cohere", model, status, "", latency_ms=latency, rate_limit=_rate_limit(headers))
    resp.error_kind = _classify(status, raw)
    if resp.error_kind:
        resp.error_detail = raw.decode(errors="replace")[:300]
        return resp
    try:
        payload = json.loads(raw)
        resp.text = "".join(c.get("text", "") for c in payload["message"]["content"])
        resp.truncated = payload.get("finish_reason") == "MAX_TOKENS"
        tokens = (payload.get("usage") or {}).get("tokens") or {}
        resp.input_tokens = int(tokens.get("input_tokens") or 0)
        resp.output_tokens = int(tokens.get("output_tokens") or 0)
    except Exception as e:
        resp.error_kind = "client_parse"
        resp.error_detail = f"{e}: {raw[:200]!r}"
    return resp


def call(provider, model, messages, temperature=0.7, max_tokens=1024, timeout=120, seed=None, thinking_level="low", reasoning=None, extra=None):
    if provider == "gemini":
        return _gemini_call(model, messages, temperature, max_tokens, timeout, thinking_level)
    if provider == "cohere":
        return _cohere_call(model, messages, temperature, max_tokens, timeout)
    if provider in OPENAI_COMPATIBLE:
        return _openai_call(provider, model, messages, temperature, max_tokens, timeout, seed, reasoning, extra)
    raise ValueError(f"unknown provider {provider}")


class Pacer:
    def __init__(self, rpm_by_key, tpm_by_key=None):
        self.min_gap = {k: 60.0 / v for k, v in rpm_by_key.items()}
        self.tpm = dict(tpm_by_key or {})
        self.last = {}
        self.token_window = {}

    def _prune(self, key, now):
        window = self.token_window.setdefault(key, [])
        while window and window[0][0] < now - 60.0:
            window.pop(0)
        return window

    def wait(self, key, provider, est_tokens=0):
        gap = self.min_gap.get(key, self.min_gap.get(provider, 0))
        now = time.time()
        due = self.last.get(key, 0) + gap
        if due > now:
            time.sleep(due - now)
        budget = self.tpm.get(key, self.tpm.get(provider))
        if budget:
            while True:
                now = time.time()
                window = self._prune(key, now)
                used = sum(t for _, t in window)
                if used + est_tokens <= budget or not window:
                    break
                time.sleep(max(0.2, window[0][0] + 60.0 - now))
        self.last[key] = time.time()

    def record(self, key, tokens):
        self.token_window.setdefault(key, []).append((time.time(), tokens))

    def backoff(self, key, resp, attempt):
        delay = None
        retry_after = resp.rate_limit.get("retry-after")
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            m = re.search(r"try again in ([0-9.]+)(ms|s)", resp.error_detail or "")
            if m:
                delay = float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
        if delay is None:
            delay = min(60.0, 2.0 * (2 ** attempt))
        time.sleep(min(90.0, delay + 0.5))


def estimate_tokens(messages, expected_output=400):
    chars = sum(len(m.get("content", "")) for m in messages)
    return int(chars / 3.2) + expected_output
