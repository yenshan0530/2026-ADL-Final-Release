"""OpenAI-compatible model client, ported from the TA engine (ADL-26W `ledger.runtime.client`).

Responsibilities that belong here and nowhere else:
  * classifying a failure as ``agent``-attributable or ``infrastructure``-attributable,
    because that single field decides whether an agent is charged a turn;
  * bounded retry of shared-infrastructure failures;
  * token accounting normalised so ``total == input + output + reasoning`` exactly, and
    ``cached_input <= input``.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

BASE_URL = "https://integrate.api.nvidia.com/v1"

# A Qwen-style function call written into the message content instead of `tool_calls`.
_TOOL_CALL_TAG = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _resolve_default_base_url() -> str:
    """Shared default endpoint: `NVIDIA_BASE_URL` wins, else the NVIDIA endpoint."""
    return (os.environ.get("NVIDIA_BASE_URL") or BASE_URL).strip()

# Failures we treat as shared infrastructure: retried under the same turn_id, never charged.
_INFRA_MARKERS = (
    "503", "502", "504", "overloaded", "temporarily", "connection", "reset",
    "timed out", "timeout", "rate limit", "429", "too many requests",
    "internal server error", "bad gateway", "service unavailable", "apiconnection",
)


def _env_or_dotenv(name: str) -> Optional[str]:
    """Value of `name` from the environment or a local .env; empty values count as unset."""
    key = os.environ.get(name)
    if key and key.strip():
        return key.strip()
    for candidate in (Path(".env"), Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
    return None


def load_api_key() -> str:
    """Key comes from the environment or a local .env — never from source."""
    key = _env_or_dotenv("NVIDIA_API_KEY")
    if key:
        return key
    # Local vLLM servers accept any key; hosted endpoints get theirs per model
    # through `api_key_env` in the model registry.
    return "EMPTY"


# OpenAI reasoning models (gpt-5*, o1/o3/o4*) reject Chat Completions sampler
# params: `max_tokens` is renamed `max_completion_tokens`, and `temperature`,
# `top_p`, `seed` and `stop` are refused outright with a 400 that this client
# would bill to the agent. The payload is adapted per model family
# instead. On OpenAI's own endpoint these models are served through the Responses
# API instead (the gpt-5.6-luna house call): the effort rides in
# `reasoning={"effort", "mode", "summary"}` and tool schemas flatten into
# Responses function tools. Either way stop sequences cannot be
# enforced server-side; the text-side `stop_sequence_hit` check in `complete`
# still applies.
_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def is_openai_reasoning_model(model: str) -> bool:
    """True when `model` is an OpenAI reasoning model that needs the adapted payload."""
    return model.startswith(_REASONING_MODEL_PREFIXES)


@dataclass
class ToolCall:
    """One native function call from the model (`message.tool_calls[0]`)."""

    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    arguments_error: Optional[str] = None   # non-None when `arguments` is not valid JSON


@dataclass
class CallResult:
    text: str
    reasoning: Optional[str]
    stop_reason: str
    stop_sequence_hit: Optional[str]
    usage: dict[str, int]
    timing: dict[str, int]
    outcome: str                       # accepted | infra_error | timeout
    error: Optional[str] = None
    attempts_infra: list[dict[str, Any]] = field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    # Responses API only: opaque reasoning items (`id` + `encrypted_content`) kept
    # for verbatim replay in a later `input` — the summary text alone is not state
    # the model can resume from.
    reasoning_items: list[dict[str, Any]] = field(default_factory=list)


class ModelClient:
    """Thin, synchronous wrapper.  Concurrency is the scheduler's job, not the client's.

    Rate limits on this endpoint are **per model**, not per account (measured 2026-09-07:
    `minimaxai/minimax-m3` returned 429 for every request, including sequential ones, while
    four other models on the same key answered normally). A wave that activates eight agents
    all bound to one model therefore looks to the endpoint like a burst against a single
    deployment. `min_interval` serialises requests per model id so one model cannot be
    exhausted by our own scheduler, which is cheaper than losing the activation and far
    cheaper than losing the run.

    Multi-endpoint: one `ModelClient` serves every LLM definition in the run. The
    YAML config may give each role (policy / TA buyer / oracle) — and, via
    `model_base_urls`, each model id — its own OpenAI-compatible `base_url`.
    `complete(..., base_url=...)` routes to the matching endpoint, lazily building
    one underlying `OpenAI` handle per distinct URL. The throttle gate is keyed on
    `(base_url, model)` so two different endpoints never share a gate.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 90.0, retry_limit: int = 2, min_interval: float = 1.2,
                 template_effort_models: Any = (),
                 endpoint_api_keys: Optional[dict[str, str]] = None):
        from openai import OpenAI
        self._openai_factory = OpenAI
        # Non-OpenAI models that take reasoning effort through chat_template_kwargs.
        # Empty by default, so a chat model never receives an effort it was not given.
        self._template_effort_models = frozenset(template_effort_models)
        # {base_url: key} for endpoints that take their own key — used by every agent on
        # that endpoint without a per-policy key, TA buyers included.
        self._endpoint_api_keys = {url.strip().rstrip("/"): key
                                   for url, key in (endpoint_api_keys or {}).items()}
        self._api_key = api_key or load_api_key()
        self._default_base_url = (base_url or _resolve_default_base_url()).strip()
        self._openai_clients: dict[str, Any] = {}
        # Stored under the same `url\x00key` cache key `_openai_for` looks up, so the
        # default endpoint gets exactly one handle instead of a silent duplicate.
        self._openai_clients[f"{self._default_base_url}\x00{self._api_key}"] = OpenAI(
            api_key=self._api_key, base_url=self._default_base_url,
            timeout=timeout, max_retries=0)
        self.timeout = timeout
        self.retry_limit = retry_limit
        self.min_interval = min_interval
        self._gates: dict[str, threading.Lock] = {}
        self._last_call: dict[str, float] = {}
        self._gate_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """Default endpoint — the one used when a call carries no explicit base_url."""
        return self._default_base_url

    def _openai_for(self, base_url: Optional[str], api_key: Optional[str] = None) -> Any:
        """One cached `OpenAI` handle per distinct (endpoint URL, API key)."""
        url = (base_url or self._default_base_url).strip()
        key = api_key or self._default_key_for_endpoint(url)
        cache_key = f"{url}\x00{key}"
        client = self._openai_clients.get(cache_key)
        if client is None:
            client = self._openai_factory(
                api_key=key, base_url=url,
                timeout=self.timeout, max_retries=0)
            self._openai_clients[cache_key] = client
        return client

    def _default_key_for_endpoint(self, url: str) -> str:
        """Default key for one endpoint. `api.openai.com` prefers `OPENAI_API_KEY`.

        TA buyers and padded baseline agents carry no per-policy api_key, so a run
        pointed at OpenAI would otherwise authenticate with the NVIDIA key and turn
        each of their turns into a 401 billed to the agent. An endpoint listed in
        `endpoint_api_keys` uses its own key; any other (NVIDIA Integrate, local vLLM, ...)
        keeps the constructor default.
        """
        listed = self._endpoint_api_keys.get(url.rstrip("/"))
        if listed:
            return listed
        if "api.openai.com" in url:
            key = _env_or_dotenv("OPENAI_API_KEY")
            if key:
                return key
        return self._api_key

    def _throttle(self, model: str, base_url: Optional[str] = None) -> None:
        """Hold a per-(endpoint, model) gate so two requests to one deployment cannot overlap."""
        if self.min_interval <= 0:
            return
        key = f"{(base_url or self._default_base_url).strip()}\x00{model}"
        with self._gate_lock:
            gate = self._gates.setdefault(key, threading.Lock())
        gate.acquire()
        try:
            wait = self.min_interval - (time.time() - self._last_call.get(key, 0.0))
            if wait > 0:
                time.sleep(wait)
        finally:
            self._last_call[key] = time.time()
            gate.release()

    @staticmethod
    def _classify(exc: BaseException) -> str:
        blob = f"{type(exc).__name__} {exc}".lower()
        return "infrastructure" if any(m in blob for m in _INFRA_MARKERS) else "agent"

    def complete(self, *, model: str, messages: list[dict[str, str]], max_tokens: int,
                 temperature: float = 0.7, top_p: float = 0.95, seed: Optional[int] = None,
                 stop: Optional[list[str]] = None,
                 reasoning_effort: Optional[str] = None,
                 tools: Optional[list[dict[str, Any]]] = None,
                 tool_choice: str = "auto",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 extra_body: Optional[dict[str, Any]] = None) -> CallResult:
        """One model call with bounded infrastructure retry.

        Retries here are *infrastructure* retries: they never produce a second turn record
        and never consume a second activation.

        When `tools` (Chat Completions function schemas, see
        `ledger.protocol.actions.to_openai_tools`) is given, the action arrives as a
        native `message.tool_calls[0]` instead of JSON fished out of prose, and
        `CallResult.tool_call` carries it. `tool_choice` defaults to `"auto"`: this
        endpoint's gateway rejects `"required"` with a 400 (measured 2026-09-08), and
        the system prompt already instructs exactly one call, with the text-envelope
        parse as fallback for a model that answers in prose anyway.

        `base_url` selects the endpoint for this call (per-LLM-definition override
        from the YAML config). `api_key` overrides the client's default key for this
        call only, allowing heterogeneous endpoints with different credentials.
        OpenAI reasoning models (gpt-5*, o*) get an adapted payload and no sampler
        params (`temperature`/`top_p`/`seed`/`stop` are rejected with a 400). On
        api.openai.com they are served through the Responses API, where the hidden
        reasoning budget arrives as `reasoning={"effort", "mode": "standard",
        "summary": "auto"}`; on every other endpoint it is sent as Chat Completions'
        `reasoning_effort` ("minimal".."xhigh"). None keeps the provider default.
        """
        endpoint = (base_url or self._default_base_url).strip()
        openai_client = self._openai_for(endpoint, api_key)
        # OpenAI's own endpoint serves its reasoning models through the Responses
        # API (the gpt-5.6-luna house call); everywhere else keeps Chat Completions.
        use_responses = is_openai_reasoning_model(model) and "api.openai.com" in endpoint
        infra_attempts: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        for attempt in range(self.retry_limit + 1):
            self._throttle(model, endpoint)
            started = time.time()
            try:
                if use_responses:
                    resp = openai_client.responses.create(**self._responses_kwargs(
                        model=model, messages=messages, max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort, tools=tools,
                        tool_choice=tool_choice))
                    wall_ms = int((time.time() - started) * 1000)
                    return self._responses_result(resp, stop, wall_ms, infra_attempts)
                # Reasoning items interleaved into `messages` for carryover are
                # Responses-API input items; Chat Completions must never see them.
                chat_messages = [m for m in messages if "role" in m]
                kwargs: dict[str, Any] = {"model": model, "messages": chat_messages}
                if is_openai_reasoning_model(model):
                    # gpt-5*/o* refuse `max_tokens` (renamed) and every sampler param.
                    kwargs["max_completion_tokens"] = max_tokens
                    if reasoning_effort:
                        kwargs["reasoning_effort"] = reasoning_effort
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["temperature"] = temperature
                    kwargs["top_p"] = top_p
                    if seed is not None:
                        kwargs["seed"] = seed
                    if stop:
                        kwargs["stop"] = stop
                    if reasoning_effort and model in self._template_effort_models:
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {"reasoning_effort": reasoning_effort}}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice
                if extra_body:
                    # Provider-specific request fields, e.g. Qwen's
                    # {"chat_template_kwargs": {"enable_thinking": false}}.
                    kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
                resp = openai_client.chat.completions.create(**kwargs)
                wall_ms = int((time.time() - started) * 1000)
                choice = resp.choices[0]
                msg = choice.message
                text = (msg.content or "").strip()
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                usage = self._normalise_usage(resp)
                finish = choice.finish_reason or "stop"
                stop_reason = {"stop": "stop", "length": "length",
                               "tool_calls": "tool_calls", "content_filter": "error"}.get(finish, finish)
                if (stop_reason in ("tool_calls", "stop") and max_tokens
                        and usage.get("output_tokens", 0) + usage.get("reasoning_tokens", 0)
                        >= max_tokens):
                    # Some servers (vLLM tool parsers) close a call cut off at max_tokens and
                    # report "tool_calls": its arguments end mid-text. Call it what it is.
                    stop_reason = "length"
                hit = None
                if stop:
                    for s in stop:
                        if text.endswith(s):
                            hit, stop_reason = s, "stop_sequence"
                tool_call = self._extract_tool_call(msg)
                if tool_call is not None and not getattr(msg, "tool_calls", None):
                    # The call came out of a `<tool_call>` tag in the content: the tag is
                    # the call, not prose, and must not be delivered as reply text.
                    text = _TOOL_CALL_TAG.sub("", text).strip()
                return CallResult(
                    text=text, reasoning=reasoning, stop_reason=stop_reason,
                    stop_sequence_hit=hit, usage=usage,
                    timing={"queued_ms": 0, "ttft_ms": 0, "wall_ms": wall_ms},
                    outcome="accepted", attempts_infra=infra_attempts,
                    tool_call=tool_call,
                )
            except BaseException as exc:                       # noqa: BLE001 — classified below
                wall_ms = int((time.time() - started) * 1000)
                kind = self._classify(exc)
                blob = f"{type(exc).__name__} {exc}".lower()
                rate_limited = "429" in blob or "too many requests" in blob or "rate limit" in blob
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                infra_attempts.append({
                    "attributable_to": "infrastructure" if kind == "infrastructure" else "agent",
                    "retry_index": attempt, "detail": last_error, "wall_ms": wall_ms,
                })
                if kind != "infrastructure" or attempt >= self.retry_limit:
                    outcome = "timeout" if "timeout" in last_error.lower() else "infra_error"
                    return CallResult(
                        text="", reasoning=None, stop_reason="error", stop_sequence_hit=None,
                        usage={"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                               "reasoning_tokens": 0, "total_tokens": 0},
                        timing={"queued_ms": 0, "ttft_ms": 0, "wall_ms": wall_ms},
                        outcome=outcome, error=last_error, attempts_infra=infra_attempts,
                    )
                # A 429 on shared hosted capacity clears on the order of seconds, not
                # milliseconds; backing off too fast just burns the retry budget and turns a
                # transient limit into a lost activation.
                base = 6.0 if rate_limited else 1.5
                cap = 45.0 if rate_limited else 8.0
                time.sleep(min(cap, base * (2 ** attempt)) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    # ------------------------------------------------------------ responses ---

    # The documented gpt-5.6-luna call: encrypted reasoning comes back for later
    # replay, and the web-search include is inert unless a web-search tool is attached.
    # `store` stays False: the engine replays reasoning items explicitly, so nothing
    # needs the response persisted server-side (the ZDR-compatible pairing).
    _RESPONSES_INCLUDE = ("reasoning.encrypted_content", "web_search_call.action.sources")

    @staticmethod
    def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chat-style thread -> Responses input items.

        An assistant tool call becomes a `function_call` item and its tool result a
        `function_call_output` item, both keyed by the call id.  Reasoning items are already
        input items and pass through in place, so each sits before the call it produced.
        """
        items: list[dict[str, Any]] = []
        for msg in messages:
            if "role" not in msg:
                items.append(msg)
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                if msg.get("content"):
                    items.append({"role": "assistant", "content": msg["content"]})
                for call in msg["tool_calls"]:
                    items.append({"type": "function_call", "call_id": call["id"],
                                  "name": call["function"]["name"],
                                  "arguments": call["function"]["arguments"]})
            elif msg["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": msg["tool_call_id"],
                              "output": msg["content"]})
            else:
                items.append({"role": msg["role"], "content": msg.get("content") or ""})
        return items

    @classmethod
    def _responses_kwargs(cls, *, model: str, messages: list[dict[str, str]],
                          max_tokens: int, reasoning_effort: Optional[str],
                          tools: Optional[list[dict[str, Any]]],
                          tool_choice: str) -> dict[str, Any]:
        """Payload for `client.responses.create` on api.openai.com (gpt-5.6-luna style).

        Sampler params do not exist on this API; the effort rides in `reasoning`
        together with the published `mode`/`summary`, and Chat Completions' nested
        function schemas flatten into Responses function tools. `reasoning` is
        omitted entirely when no effort is configured, keeping the provider default.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "input": cls._responses_input(messages),
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "store": False,
            "include": list(cls._RESPONSES_INCLUDE),
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort,
                                  "mode": "standard", "summary": "auto"}
        if tools:
            flat = []
            for tool in tools:
                if tool.get("type") != "function":
                    continue
                fn = tool.get("function") or {}
                flat.append({
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters")
                                  or {"type": "object", "properties": {}},
                    "strict": False,
                })
            kwargs["tools"] = flat
            kwargs["tool_choice"] = tool_choice
        return kwargs

    @staticmethod
    def _responses_result(resp: Any, stop: Optional[list[str]], wall_ms: int,
                          infra_attempts: list[dict[str, Any]]) -> CallResult:
        """Fold one Responses API object into the engine's CallResult shape.

        Output items keep their native order: `reasoning` items contribute the
        captured summary, the first `function_call` becomes the tool call, and
        `output_text` is the visible prose (often empty when a call was made).
        """
        text = (getattr(resp, "output_text", None) or "").strip()
        reasoning_parts: list[str] = []
        reasoning_items: list[dict[str, Any]] = []
        tool_call: Optional[ToolCall] = None
        for item in getattr(resp, "output", None) or []:
            kind = getattr(item, "type", None)
            if kind == "reasoning":
                reasoning_items.append(ModelClient._reasoning_item_param(item))
                for summary in getattr(item, "summary", None) or []:
                    chunk = getattr(summary, "text", None)
                    if chunk:
                        reasoning_parts.append(chunk)
            elif kind == "function_call" and tool_call is None:
                name = (getattr(item, "name", None) or "").strip()
                raw = getattr(item, "arguments", "") or ""
                tool_call = ModelClient._tool_call_from_raw(name, raw)
        status = getattr(resp, "status", None) or "completed"
        stop_reason = {"completed": "stop", "incomplete": "length",
                       "failed": "error"}.get(status, status)
        hit = None
        if stop:                # no server-side stop support on this API either
            for s in stop:
                if text.endswith(s):
                    hit, stop_reason = s, "stop_sequence"
        return CallResult(
            text=text, reasoning="\n".join(reasoning_parts) or None,
            stop_reason=stop_reason, stop_sequence_hit=hit,
            usage=ModelClient._normalise_responses_usage(resp),
            timing={"queued_ms": 0, "ttft_ms": 0, "wall_ms": wall_ms},
            outcome="accepted", attempts_infra=infra_attempts,
            tool_call=tool_call, reasoning_items=reasoning_items,
        )

    @staticmethod
    def _reasoning_item_param(item: Any) -> dict[str, Any]:
        """One Responses reasoning item shaped for verbatim replay in a later `input`.

        Keeps only the fields `ResponseReasoningItemParam` accepts; the encrypted
        blob is what carries the hidden chain when `store` is False.
        """
        def texts(seq: Any, kind: str) -> list[dict[str, str]]:
            out = []
            for s in seq or []:
                text = s.get("text") if isinstance(s, dict) else getattr(s, "text", None)
                stype = (s.get("type") if isinstance(s, dict)
                         else getattr(s, "type", None)) or kind
                if text:
                    out.append({"type": stype, "text": text})
            return out

        dump = (item.model_dump() if hasattr(item, "model_dump")
                else {k: getattr(item, k, None) for k in
                      ("id", "summary", "encrypted_content", "status", "content")})
        param: dict[str, Any] = {
            "id": dump.get("id") or "",
            "type": "reasoning",
            "summary": texts(dump.get("summary"), "summary_text"),
        }
        content = texts(dump.get("content"), "reasoning_text")
        if content:
            param["content"] = content
        for key in ("encrypted_content", "status"):
            if dump.get(key):
                param[key] = dump[key]
        return param

    @staticmethod
    def _normalise_responses_usage(resp: Any) -> dict[str, int]:
        """Responses API usage into the identity above.

        Same convention as Chat Completions: `output_tokens` counts visible text
        only, with `output_tokens_details.reasoning_tokens` peeled out, and
        `cached_input <= input` (`input_tokens_details.cached_tokens`).
        """
        usage = getattr(resp, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        reasoning = 0
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
        cached = 0
        pdetails = getattr(usage, "input_tokens_details", None)
        if pdetails is not None:
            cached = int(getattr(pdetails, "cached_tokens", 0) or 0)
        out = max(0, out - reasoning)          # vendor counted reasoning inside output
        cached = min(cached, inp)
        return {
            "input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out,
            "reasoning_tokens": reasoning, "total_tokens": inp + out + reasoning,
        }

    @staticmethod
    def _extract_tool_call(msg: Any) -> Optional[ToolCall]:
        """First native function call on the message, or a Qwen-style <tool_call> tag."""
        import json as _json
        import re
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            text = (getattr(msg, "content", None) or "").strip()
            if text:
                # Qwen-3 often emits tool calls wrapped in <tool_call> XML tags inside content.
                match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
                if match:
                    try:
                        parsed = _json.loads(match.group(1))
                        if isinstance(parsed, dict):
                            name = (parsed.get("name") or "").strip()
                            arguments = parsed.get("arguments", {})
                            if not isinstance(arguments, dict):
                                arguments = {}
                            raw = _json.dumps(arguments, ensure_ascii=False)
                            return ToolCall(name=name, arguments=arguments, raw_arguments=raw)
                    except Exception:                        # noqa: BLE001
                        pass
                    # Qwen3-Coder style: <function=name><parameter=k>v</parameter></function>
                    xml = re.search(r"<function=(?P<name>[^>\s]+)\s*>(?P<body>.*?)</function>",
                                    match.group(1), re.DOTALL)
                    if xml:
                        arguments = {}
                        for p in re.finditer(
                                r"<parameter=(?P<key>[^>\s]+)\s*>(?P<value>.*?)</parameter>",
                                xml.group("body"), re.DOTALL):
                            value = p.group("value").strip()
                            try:
                                arguments[p.group("key")] = _json.loads(value)
                            except Exception:                # noqa: BLE001
                                arguments[p.group("key")] = value
                        return ToolCall(name=xml.group("name").strip(), arguments=arguments,
                                        raw_arguments=_json.dumps(arguments, ensure_ascii=False))
            return None
        first = calls[0]
        fn = getattr(first, "function", None)
        name = (getattr(fn, "name", None) or "").strip()
        raw = getattr(fn, "arguments", "") or ""
        return ModelClient._tool_call_from_raw(name, raw)

    @staticmethod
    def _tool_call_from_raw(name: str, raw: str) -> ToolCall:
        """Parse one function call's JSON arguments; malformed JSON is reported, not raised."""
        import json as _json
        try:
            arguments = _json.loads(raw) if raw.strip() else {}
            if not isinstance(arguments, dict):
                return ToolCall(name=name, arguments={}, raw_arguments=raw[:500],
                                arguments_error="arguments_not_an_object")
            return ToolCall(name=name, arguments=arguments, raw_arguments=raw[:500])
        except Exception as exc:                            # noqa: BLE001 — reported, not raised
            return ToolCall(name=name, arguments={}, raw_arguments=raw[:500],
                            arguments_error=f"tool_arguments_not_json: {type(exc).__name__}")

    @staticmethod
    def _normalise_usage(resp: Any) -> dict[str, int]:
        """Force the vendor's numbers into the identity above.

        Providers disagree about whether reasoning tokens are inside `completion_tokens`.
        We define it: `output_tokens` is visible completion only, `reasoning_tokens` is
        hidden reasoning, `cached_input_tokens` is a subset of `input_tokens`, and
        `total == input + output + reasoning` exactly.
        """
        usage = getattr(resp, "usage", None)
        inp = int(getattr(usage, "prompt_tokens", 0) or 0)
        out = int(getattr(usage, "completion_tokens", 0) or 0)
        reasoning = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
        cached = 0
        pdetails = getattr(usage, "prompt_tokens_details", None)
        if pdetails is not None:
            cached = int(getattr(pdetails, "cached_tokens", 0) or 0)
        out = max(0, out - reasoning)          # vendor counted reasoning inside completion
        cached = min(cached, inp)
        return {
            "input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out,
            "reasoning_tokens": reasoning, "total_tokens": inp + out + reasoning,
        }
