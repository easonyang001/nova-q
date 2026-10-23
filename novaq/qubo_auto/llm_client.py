from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import itertools
import re

from openai import BadRequestError, OpenAI, RateLimitError


class TokenBudgetExceeded(Exception):
    pass


class PromptTooLong(Exception):
    """A prompt that cannot fit the model's context: a bug in how that step builds its prompt."""


class StreamTimeout(Exception):
    """A streamed answer passed its deadline (the stream is closed, so the server stops generating)."""


class ReasoningLoop(Exception):
    """The model's reasoning repeats the same passage verbatim and will not finish."""
    def __init__(self, tail: str):
        super().__init__("reasoning repeats itself")
        self.tail = tail


def looping(text: str, min_period: int = 30, max_period: int = 1500, repeats: int = 4) -> bool:
    """True when the end of `text` is one passage repeated `repeats` times verbatim. Greedy decoding (temperature 0)
    sent Gemma's reasoning on route problems into such a loop after ~2 minutes; it then ran to the 20-minute timeout
    (Paris and Melbourne model steps, 2026-10-21/22)."""
    t = text[-(max_period * repeats):]
    for p in range(min_period, min(max_period, len(t) // repeats) + 1):
        last = t[-p:]
        if all(t[-(k + 1) * p:-k * p] == last for k in range(1, repeats)):
            return True
    return False


MAX_COMPLETION_TOKENS = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", "32768"))   # thinking + answer, per call
_SEQ = itertools.count()      # unique transcript names: two calls of one step within a second used to overwrite


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish estimate: CJK characters ~1 token each, other text ~3.5 characters per token."""
    cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text))
    return cjk + int((len(text) - cjk) / 3.5) + 1


def split_model(model: str) -> tuple[str | None, str]:
    """'provider:model' -> (provider, model); a bare name -> (None, name) = the default provider (LLM_PROVIDER).
    Lets each agent use a different model or company, e.g. 'openrouter:<vendor>/<model>' for a hard step."""
    head, sep, tail = model.partition(":")
    if sep and head.lower() in _PROVIDERS:
        return head.lower(), tail
    return None, model


# provider -> (default base_url, env var holding the API key; None = no real key needed)
_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "local": ("http://localhost:8000/v1", None),  # vLLM / any OpenAI-compatible server
    "local2": ("http://localhost:8001/v1", None),  # a second local server (the independent checkers' model)
}


def resolve_endpoint(provider: str | None = None) -> tuple[str, str, str]:
    """Return (provider, base_url, api_key). Without `provider`: LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY
    (defaults to OpenRouter so the original behaviour is unchanged). With an explicit provider (from a
    'provider:model' name): that provider's URL and key; LLM_BASE_URL / LLM_API_KEY apply only to the default."""
    default = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
    provider = (provider or default).lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; expected one of {sorted(_PROVIDERS)}")
    default_url, key_env = _PROVIDERS[provider]
    is_default = provider == default
    base_url = (os.environ.get("LLM_BASE_URL") if is_default else None) or default_url
    api_key = ((os.environ.get("LLM_API_KEY") if is_default else None)
               or (os.environ.get(key_env, "") if key_env else "EMPTY"))
    if key_env and not api_key:
        raise ValueError(f"provider {provider!r} needs {key_env} in .env")
    return provider, base_url, api_key


def _no_think(model: str, prompt: str) -> str:
    """Prepend /no_think for Qwen3 models to disable chain-of-thought thinking."""
    if "qwen3" in model.lower() or "qwen3" in model.lower():
        return "/no_think\n" + prompt
    return prompt


class LLMClient:
    def __init__(self, transcript_dir: Path, run_id: str, max_tokens: int = 200_000, use_cache: bool = True):
        self.provider, base_url, api_key = resolve_endpoint()
        self.request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "180"))
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout, max_retries=0)
        self._clients: dict[str, OpenAI] = {self.provider: self.client}
        self._context: dict[str, int | None] = {}
        self.transcript_dir = transcript_dir / run_id
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self.tokens_used = 0
        self.cost_used = 0.0
        self._cache: dict[str, str] = {}
        if use_cache:
            self._load_cache()

    def _client_for(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            _, base_url, api_key = resolve_endpoint(provider)
            self._clients[provider] = OpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout, max_retries=0)
        return self._clients[provider]

    def context_length(self, provider: str, model: str) -> int | None:
        """The served context window (vLLM reports max_model_len); None when the server does not say."""
        key = f"{provider}:{model}"
        if key not in self._context:
            n = None
            try:
                for m in self._client_for(provider).models.list().data:
                    if m.id == model:
                        n = getattr(m, "max_model_len", None) or (m.model_extra or {}).get("max_model_len")
            except Exception:
                pass
            self._context[key] = int(n) if n else None
        return self._context[key]

    def _cache_path(self) -> Path:
        return self.transcript_dir.parent / ".llm_cache.json"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        path = self._cache_path()
        path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def _cache_key(self, model: str, prompt: str) -> str:
        raw = f"{model}||{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def call(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        step_name: str = "step",
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        explicit, model = split_model(model)
        provider = explicit or self.provider
        if provider == "local" and not explicit and os.environ.get("LLM_MODEL"):
            # The local server serves one model: map remote names (e.g. "qwen/qwen3.6-35b-a3b") onto it.
            model = os.environ["LLM_MODEL"]
        client = self._client_for(provider)
        prompt = _no_think(model, prompt)
        limit = self.context_length(provider, model)
        need = estimate_tokens(prompt)
        if limit and need > limit - 2048:
            raise PromptTooLong(f"step '{step_name}': prompt is ~{need} tokens but {model} accepts {limit}. "
                                f"The step must summarise its inputs instead of pasting them")

        cache_key = self._cache_key(model, prompt)
        if self.use_cache and temperature == 0.0 and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._log(step_name, model, temperature, prompt, cached, usage=None, from_cache=True)
            return cached

        if self.tokens_used >= self.max_tokens:
            raise TokenBudgetExceeded(
                f"Token budget of {self.max_tokens} exceeded (used {self.tokens_used})"
            )

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if limit:
            # the estimate runs low on code and JSON: leave a quarter of it as margin (a 32k-context server refused
            # 29511 output tokens on a prompt estimated at ~2.6k, 2026-10-12)
            kwargs["max_tokens"] = max(1024, min(max_tokens or MAX_COMPLETION_TOKENS, limit - int(need * 1.25) - 1024))
        elif max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if provider in ("local", "local2"):
            # Qwen3.6 ignores the /no_think prefix and reasons by default (as it did via OpenRouter
            # in the original runs); LLM_ENABLE_THINKING=false turns it off. vLLM's --reasoning-parser
            # keeps the reasoning out of `content`. Templates without this variable ignore it.
            # The second server (local2) has its own switch, default off: LLM_ENABLE_THINKING_LOCAL2.
            var, default = ("LLM_ENABLE_THINKING", "true") if provider == "local" else ("LLM_ENABLE_THINKING_LOCAL2", "false")
            thinking = os.environ.get(var, default).strip().lower() not in ("0", "false", "no")
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}

        thinking = bool((kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking"))
        stream = provider in ("local", "local2") and thinking
        try:
            try:
                resp = self._create(client, kwargs, step_name, stream=stream)
            except ReasoningLoop as loop:
                # keep thinking, but leave greedy decoding: the model's own sampling with a repetition penalty
                self._log(step_name + "_loop", model, temperature, prompt, "", None, reasoning=loop.tail)
                print(f"  [{step_name}] reasoning loop detected: retrying with sampling", flush=True)
                kwargs = {**kwargs, "temperature": 0.7,
                          "extra_body": {**kwargs["extra_body"], "repetition_penalty": 1.1, "top_p": 0.95}}
                resp = self._create(client, kwargs, step_name, stream=True)
                step_name += "_resampled"
        except Exception as e:
            # a thinking model that is still reasoning when the request times out (Gemma, model step, 4 x 15 min,
            # Melbourne 2026-10-21): answer once without thinking, like the exhausted-budget case below
            thinking_on = (kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking")
            if not ("Timeout" in type(e).__name__ or isinstance(e, ReasoningLoop)) or provider not in ("local", "local2") \
                    or not thinking_on:
                raise
            self._log(step_name + ("_loop" if isinstance(e, ReasoningLoop) else "_timeout"), model, temperature, prompt,
                      "", None, reasoning=getattr(e, "tail", ""))
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            resp = self._create(client, kwargs, step_name)
            step_name += "_nothink"
        content = resp.choices[0].message.content or ""
        self._count(resp.usage)
        if not content.strip() and resp.choices[0].finish_reason == "length" and provider in ("local", "local2") \
                and (kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking"):
            # spent the whole budget thinking and wrote nothing: log it, then answer once without thinking
            self._log(step_name + "_exhausted", model, temperature, prompt, "", resp.usage)
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            resp = self._create(client, kwargs, step_name)
            content = resp.choices[0].message.content or ""
            self._count(resp.usage)
            step_name += "_nothink"
        usage = resp.usage
        reasoning = getattr(resp, "reasoning", "")

        if self.use_cache and temperature == 0.0:
            self._cache[cache_key] = content
            self._save_cache()

        self._log(step_name, model, temperature, prompt, content, usage, reasoning=reasoning)
        return content

    def _open(self, client: OpenAI, kwargs: dict, stream: bool):
        if not stream:
            return client.chat.completions.create(**kwargs)
        return self._collect(client.chat.completions.create(**kwargs, stream=True,
                                                            stream_options={"include_usage": True}))

    def _collect(self, chunks):
        """Assemble a streamed answer (reasoning kept apart); stop at a reasoning loop or at the deadline."""
        if hasattr(chunks, "choices"):                 # a client that ignores stream=True (test fakes)
            return chunks
        start, parts, thought, finish, usage, seen = time.time(), [], [], None, None, 0
        try:
            for chunk in chunks:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices:
                    ch = chunk.choices[0]
                    d = ch.delta
                    extra = getattr(d, "model_extra", None) or {}
                    r = getattr(d, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
                    if r:
                        thought.append(r)
                        seen += 1
                        if seen % 40 == 0 and looping("".join(thought[-400:])):
                            raise ReasoningLoop("".join(thought)[-4000:])
                    if d.content:
                        parts.append(d.content)
                    finish = ch.finish_reason or finish
                if time.time() - start > self.request_timeout:
                    raise StreamTimeout(f"no complete answer after {self.request_timeout:.0f} s")
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
        message = SimpleNamespace(content="".join(parts))
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage,
                               reasoning="".join(thought))

    def _create(self, client: OpenAI, kwargs: dict, step_name: str, stream: bool = False):
        # Retry on 429 with exponential backoff (free-tier rate limits)
        for attempt in range(5):
            try:
                return self._open(client, kwargs, stream)
            except RateLimitError:
                if attempt == 4:
                    raise
                wait = 30 * (2 ** attempt)  # 30, 60, 120, 240 s
                print(f"  [rate limit] waiting {wait}s before retry {attempt+2}/5...", flush=True)
                time.sleep(wait)
            except BadRequestError as e:
                msg = str(e)
                if "context length" in msg or "maximum context" in msg:
                    # the token estimate runs low on some prompts (4223 real vs ~2560 estimated, Gemma, 2026-10-21):
                    # the server states both numbers, so shrink max_tokens to fit and retry once
                    lim = re.search(r"maximum context length is (\d+)", msg)
                    got = re.search(r"prompt contains at least (\d+)", msg)
                    if lim and got and kwargs.get("max_tokens"):
                        room = int(lim.group(1)) - int(got.group(1)) - 512
                        if room >= 1024:
                            kwargs = {**kwargs, "max_tokens": room}
                            print(f"  [{step_name}] prompt longer than estimated: retrying with max_tokens={room}",
                                  flush=True)
                            try:
                                return self._open(client, kwargs, stream)
                            except BadRequestError as e2:
                                raise PromptTooLong(f"step '{step_name}': {str(e2)[:300]}") from e2
                    raise PromptTooLong(f"step '{step_name}': {msg[:300]}") from e
                raise

    def _count(self, usage) -> None:
        if usage:
            self.tokens_used += usage.total_tokens
            self.cost_used += float(getattr(usage, "cost", 0.0) or 0.0)

    def _log(
        self,
        step_name: str,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
        usage,
        from_cache: bool = False,
        reasoning: str = "",
    ) -> None:
        ts = int(time.time())
        log_path = self.transcript_dir / f"{step_name}_{ts}_{next(_SEQ):04d}.json"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": response,
            "reasoning": reasoning,
            "from_cache": from_cache,
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
            "timestamp": ts,
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
