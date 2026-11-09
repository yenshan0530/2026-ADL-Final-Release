"""Policies: how an agent turns an observation into one tool call.

`LLMPolicy` calls a model endpoint (base model or LoRA). The scripted policies are
rule-based: TA opponents, baselines, and the stand-ins `--mock-llm` uses so a whole run can
be tested without a GPU or API key.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from adlmarket.config import AgentSpec, ModelSpec
from adlmarket.prompts import observation_message


@dataclass
class Decision:
    tool: Optional[str]
    args: dict[str, Any]
    private_state: Optional[str] = None
    log: dict[str, Any] = field(default_factory=dict)   # model call details for the run log
    # Without a tool: the result the agent is shown instead (why there was no valid call).
    error: Optional[dict[str, Any]] = None
    # The model call itself failed (server down, timeout, rate limit after retries): not
    # the agent's fault, so it is counted separately and kept out of the agent's history.
    infra: bool = False


class Policy:
    def decide(self, obs: dict[str, Any]) -> Decision:
        raise NotImplementedError

    def observe_result(self, result: dict[str, Any]) -> None:
        """Called with the outcome of this agent's own last action."""


# ------------------------------------------------------------------------------ LLM ---

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Every JSON object in a reply, in order: fenced blocks first, then the whole text,
    then each balanced {...} span (string-aware, so braces inside strings are skipped).
    Ported from ADL-26W's envelope parser: models wrap JSON in prose and fences."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        chunk = chunk.strip()
        try:
            parsed = json.loads(chunk)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            yield parsed
        elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            yield parsed[0]
        start = chunk.find("{")
        while start != -1:
            depth, in_str, esc, end = 0, False, False, -1
            for i in range(start, len(chunk)):
                ch = chunk[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                start = chunk.find("{", start + 1)
                continue
            try:
                parsed = json.loads(chunk[start:end + 1])
            except ValueError:
                start = chunk.find("{", start + 1)        # maybe a nested object parses
                continue
            if isinstance(parsed, dict):
                yield parsed
            start = chunk.find("{", end + 1)


def _as_call(obj: dict[str, Any]) -> Optional[tuple[str, dict[str, Any]]]:
    """{"name"|"tool"|"action": ..., "arguments"|"parameters"|"args": {...}}, optionally
    wrapped as {"function": {...}} or {"tool_call(s)": ...}."""
    for key in ("tool_call", "tool_calls", "function"):
        inner = obj.get(key)
        if isinstance(inner, list) and len(inner) == 1:
            inner = inner[0]
        if isinstance(inner, dict):
            return _as_call(inner)
    name = obj.get("name") or obj.get("tool") or obj.get("action")
    args = next((obj[k] for k in ("arguments", "parameters", "args") if k in obj), {})
    if isinstance(args, str):
        args = next(_json_objects(args), None)
    if isinstance(name, str) and name and isinstance(args, dict):
        return name, args
    return None


def parse_text_call(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Fallback for models that write the call as JSON in the reply instead of using
    native tool calls. Reasoning in <think> tags is skipped; the first object that has
    the shape of a call wins."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:                    # an opening tag the template already consumed
        text = text.rsplit("</think>", 1)[1]
    for obj in _json_objects(text):
        call = _as_call(obj)
        if call:
            return call
    return None


_LENGTH_NUDGE = ("\n\nYour previous reply to this turn ran out of tokens before it finished "
                 "a function call. Think briefly, then call exactly one function.")


# A call gets at least this many output tokens; with less left of the round's budget the
# agent waits instead of making a call that could not finish anything.
MIN_CALL_TOKENS = 64


def _generated(usage: Optional[dict[str, Any]]) -> int:
    """Tokens a reply generated: visible text plus thinking (counted apart by the client)."""
    u = usage or {}
    return int(u.get("output_tokens") or 0) + int(u.get("reasoning_tokens") or 0)


def _context_overflow(error: Optional[str]) -> bool:
    """The server rejected the prompt as longer than its context (vLLM / OpenAI wording)."""
    blob = (error or "").lower()
    return "context length" in blob or "context_length_exceeded" in blob


def _without_board(user: dict[str, Any]) -> dict[str, Any]:
    """A past observation message without its board (the current turn shows the board)."""
    try:
        body = json.loads(user["content"])
    except (TypeError, ValueError):
        return user
    if "board" not in body:
        return user
    body["board"] = "(omitted in past turns; see the current board)"
    return {**user, "content": json.dumps(body, ensure_ascii=False)}


class LLMPolicy(Policy):
    def __init__(self, agent: AgentSpec, model: ModelSpec, client: Any, system: str,
                 tools: list[dict[str, Any]], history_turns: int, state_chars: int,
                 seed: Optional[int] = None, save_prompts: bool = False,
                 history_board: str = "full"):
        self.agent = agent
        self.model = model
        self.client = client
        self.system = system
        self.tools = tools
        self.history_turns = history_turns
        self.state_chars = state_chars
        self.seed = seed
        self.save_prompts = save_prompts
        self.history_board = history_board
        # A LoRA is pinned to one replica by its name, a base-model agent by its id.
        self.base_url = model.url_for(model.model if model.lora else agent.id)
        self.private_state = ""
        # Output tokens this agent may still generate this round (rounds.output_tokens_per_round);
        # set by the engine before each turn. None: no limit.
        self.budget: Optional[int] = None
        # History turns that fit the context on the last call: the next call starts one above,
        # so a long history costs at most one rejected request per turn, not one per turn cut.
        self._fit: Optional[int] = None
        self.history: list[list[dict[str, Any]]] = []    # one message list per past turn
        self._pending: list[dict[str, Any]] = []
        self._pending_call_id: Optional[str] = None
        self._last_result: Optional[dict[str, Any]] = None
        self._skipped = False
        self._turn = 0

    def decide(self, obs: dict[str, Any]) -> Decision:
        self._turn += 1
        user = {"role": "user",
                "content": observation_message(obs, self.private_state, self._last_result)}
        # Show the last `history_turns` turns. If the prompt is over the server's context
        # length, drop the oldest turns until it fits rather than failing every turn. Without
        # any history the prompt always fits (system prompt + one observation + the reply).
        keep = min(self.history_turns, len(self.history))
        if self._fit is not None:
            keep = min(keep, self._fit + 1)
        cut = 0
        prompt, truncated = user, []
        spent = 0                                # output tokens generated this turn (all calls)
        while True:
            past = [m for turn in self.history[len(self.history) - keep:] for m in turn]
            messages = [{"role": "system", "content": self.system}, *past, prompt]
            limit = self.model.max_tokens
            if self.budget is not None:          # the round's output budget caps every call
                limit = min(limit, self.budget - spent)
            res = self.client.complete(
                model=self.model.model, messages=messages, max_tokens=limit,
                temperature=self.model.temperature, top_p=self.model.top_p, seed=self.seed,
                reasoning_effort=self.model.reasoning_effort, tools=self.tools,
                base_url=self.base_url, api_key=self.model.resolve_key(),
                extra_body=self.model.extra_body or None)
            spent += _generated(res.usage)
            if keep > 0 and res.outcome != "accepted" and _context_overflow(res.error):
                keep -= 1
                cut += 1
                continue
            # A reply cut off at max_tokens wastes the agent's turn: ask again, once or
            # `length_retries` times, with a nudge (which also changes the prompt, so a
            # fixed seed does not replay the same cut-off reply).
            if (res.outcome == "accepted" and res.stop_reason == "length"
                    and len(truncated) < self.model.length_retries
                    and (self.budget is None or self.budget - spent >= MIN_CALL_TOKENS)):
                truncated.append({"usage": res.usage, "timing": res.timing})
                prompt = {**user, "content": user["content"] + _LENGTH_NUDGE}
                continue
            break
        log = {"model": self.model.model, "base_url": self.base_url,
               "outcome": res.outcome, "error": res.error, "stop_reason": res.stop_reason,
               "text": res.text, "reasoning": res.reasoning, "usage": res.usage,
               "timing": res.timing, "attempts_infra": res.attempts_infra,
               "prompt_messages": len(messages), "history_turns_shown": keep,
               "output_tokens_spent": spent, "max_tokens": limit}
        if cut:
            log["context_retries"] = cut         # requests rejected as over the context
        if not _context_overflow(res.error):
            self._fit = keep
        if truncated:
            log["length_retries"] = truncated
        if self.save_prompts:
            log["messages"] = messages

        if res.outcome != "accepted":
            # Nothing reached the agent: leave its history as it was.
            self._pending, self._pending_call_id, self._skipped = [], None, True
            return Decision(None, {}, private_state=self.private_state, log=log, infra=True,
                            error={"ok": False, "error": "model_call_failed",
                                   "detail": "the model server did not answer this turn "
                                             "(not counted against you)"})
        self._skipped = False

        tool, args, error = None, {}, None
        call = res.tool_call
        if res.stop_reason == "length":
            # Never act on a reply cut off at max_tokens: a call in it may be missing the
            # end of its arguments (a message stopping mid-word, a wrong qty).
            if call is not None:
                log["truncated_call"] = call.name
            error = {"ok": False, "error": "no_tool_call",
                     "detail": "your reply hit the max_tokens limit before it finished a "
                               "function call; think less and call one function"}
        elif call is not None:
            log["raw_arguments"] = call.raw_arguments
            if not call.arguments_error:
                tool, args = call.name, dict(call.arguments)
            else:
                repaired = next(_json_objects(call.raw_arguments or ""), None)
                if repaired is not None:
                    tool, args = call.name, repaired
                    log["arguments_repaired"] = True
                else:
                    log["parse_error"] = call.arguments_error
                    error = {"ok": False, "error": "bad_tool_call",
                             "detail": f"the arguments of {call.name!r} were not a JSON object "
                                       f"({call.arguments_error})"}
        else:
            parsed = parse_text_call(res.text)
            if parsed:
                tool, args = parsed
                log["parsed_from_text"] = True
            else:
                error = {"ok": False, "error": "no_tool_call",
                         "detail": "your reply contained no function call; respond with "
                                   "exactly one function call"}

        state = args.pop("private_state", None) if tool else None
        if state is not None and not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        if isinstance(state, str):
            self.private_state = state[: self.state_chars]

        # Responses API reasoning items go back in before the turn they produced, so the
        # model resumes its own reasoning chain (ADL-26W formatter.render_messages).
        self._pending = [user, *res.reasoning_items]
        if tool:
            self._pending_call_id = f"call_{self._turn}"
            call_args = dict(args)
            if state is not None:
                call_args["private_state"] = state
            self._pending.append({"role": "assistant", "content": res.text or None, "tool_calls": [{
                "id": self._pending_call_id, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(call_args, ensure_ascii=False)}}]})
        else:
            self._pending_call_id = None
            self._pending.append({"role": "assistant", "content": res.text or ""})
        return Decision(tool=tool, args=args, private_state=self.private_state, log=log,
                        error=error)

    def observe_result(self, result: dict[str, Any]) -> None:
        if self._skipped:
            return
        if self._pending_call_id:
            self._pending.append({"role": "tool", "tool_call_id": self._pending_call_id,
                                  "content": json.dumps(result, ensure_ascii=False)})
            self._last_result = None
        else:
            self._last_result = result
        if self.history_board == "latest" and self._pending:
            self._pending[0] = _without_board(self._pending[0])
        self.history.append(self._pending)
        self._pending = []


# ------------------------------------------------------------------------- scripted ---

_PRICE = re.compile(r"\$?\s*(\d+(?:\.\d{1,2})?)")


def _mentioned_price(text: str) -> Optional[float]:
    for m in _PRICE.finditer(text):
        if "$" in m.group(0) or "." in m.group(1):
            return float(m.group(1))
    return None


def _mentioned_qty(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:x|units?|pcs|pieces)\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


class ScriptedSeller(Policy):
    """List near the list price, adjust each round by sales, haggle in DMs.

    params: list_factor (1.25 x list price), step (0.10), min_markup (1.15 x cost),
            concede (0.5: counter halfway between the ask and the current price),
            cheat (0.0: chance, per quality-varying product, of delivering low grade while
            pricing as high), certify (true: certify high-grade units when the market allows).
    """

    def __init__(self, agent: AgentSpec, situation: dict[str, Any], spec: Any):
        p = agent.params
        rng = random.Random(f"seller:{agent.id}")
        self.id = agent.id
        self.step = p.get("step", 0.10)
        self.concede = p.get("concede", 0.5)
        self.cost = situation["your_unit_cost"]
        self.list_price = {x["product_id"]: x["list_price"] for x in situation["products"]}
        floor_markup = p.get("min_markup", 1.15)
        enforced = situation.get("your_minimum_price", {})
        self.floor = {pid: max(round(c * floor_markup, 2), enforced.get(pid, 0))
                      for pid, c in self.cost.items()}
        self.price = {pid: max(round(self.list_price[pid] * p.get("list_factor", 1.25), 2),
                               self.floor[pid]) for pid in self.cost}
        varies = situation.get("your_low_grade_unit_cost", {})
        self.grade = {pid: "low" if rng.random() < p.get("cheat", 0.0) else "high" for pid in varies}
        self.certify = p.get("certify", True) and spec.schema_.quality.certification
        self.round = 0
        self.sold_this_round: dict[str, int] = {}
        self.queue: list[tuple[str, dict[str, Any]]] = []
        self.listed: set[str] = set()

    def _quality_args(self, pid: str) -> dict[str, Any]:
        if pid not in self.grade:
            return {}
        args: dict[str, Any] = {"grade": self.grade[pid]}
        if self.certify and self.grade[pid] == "high":
            args["certified"] = True
        return args

    def decide(self, obs: dict[str, Any]) -> Decision:
        stock = obs["you"]["stock"]
        if obs["round"] != self.round:
            if self.round:                       # reprice from last round's sales
                for pid in self.price:
                    sold = self.sold_this_round.get(pid, 0)
                    factor = 1 + self.step if sold >= 2 else (1 - self.step if sold == 0 else 1)
                    new = max(self.floor[pid], round(self.price[pid] * factor, 2))
                    if new != self.price[pid]:
                        self.price[pid] = new
                        self.listed.discard(pid)
            self.round, self.sold_this_round = obs["round"], {}
        for m in obs["new_messages"]:
            if m["channel"] == "notice" and m["text"].startswith("Trade") and f"from {self.id} " in m["text"]:
                pid = m["text"].split(" x ")[1].split(" ")[0]
                qty = int(m["text"].split(" bought ")[1].split(" x ")[0])
                self.sold_this_round[pid] = self.sold_this_round.get(pid, 0) + qty
            elif m["channel"] == "dm" and m["from"] != self.id:
                self.queue.append(self._answer(m, stock))
        for pid in self.price:                   # list every product before haggling
            if pid not in self.listed and stock.get(pid, 0) > 0:
                self.listed.add(pid)
                return Decision("set_price", {"product_id": pid, "price": self.price[pid],
                                              **self._quality_args(pid)})
        while self.queue:
            tool, args = self.queue.pop(0)
            if tool == "make_offer" and stock.get(args["product_id"], 0) < args["qty"]:
                continue
            return Decision(tool, args)
        return Decision("wait", {})

    def _answer(self, m: dict[str, Any], stock: dict[str, int]) -> tuple[str, dict[str, Any]]:
        text = m["text"]
        pid = next((p for p in self.price if p in text), None) \
            or next((p for p, q in stock.items() if q > 0), None)
        if pid is None or stock.get(pid, 0) == 0:
            return "send_message", {"to": m["from"], "message": "Sorry, I am sold out."}
        qty = min(stock[pid], _mentioned_qty(text) or 1)
        ask = _mentioned_price(text)
        current = self.price[pid]
        if ask is None:
            price = current
        elif ask >= self.floor[pid]:
            price = max(ask, round(current - (current - ask) * (1 - self.concede), 2))
        else:
            price = max(self.floor[pid], round(ask + (current - ask) * self.concede, 2))
        price = min(price, current)
        return "make_offer", {"buyer_id": m["from"], "product_id": pid, "price": price,
                              "qty": qty, "message": "Here is my best price for you.",
                              **self._quality_args(pid)}


class ScriptedBuyer(Policy):
    """Buy the units with the best utility for the price, one at a time; haggle otherwise.

    params: target (0.80: aim for this x the unit's expected utility), accept (0.95: take
            an offer up to this x), panic_round (pay up to full expected utility from this
            round on; default = min rounds, since the market may close after that),
            p_low (0.3: assumed chance an uncertified quality-varying unit is low grade),
            review (true: review every trade, 1 star for low grade, 5 for high),
            avoid_rating (3.0: treat sellers rated below this as delivering low grade).
    """

    def __init__(self, agent: AgentSpec, situation: dict[str, Any], spec: Any):
        p = agent.params
        self.id = agent.id
        self.target = p.get("target", 0.80)
        self.accept = p.get("accept", 0.95)
        self.panic_round = p.get("panic_round", spec.rounds.total)
        self.p_low = p.get("p_low", 0.3)
        self.review = p.get("review", True)
        self.avoid_rating = p.get("avoid_rating", 3.0)
        self.low_utility = spec.schema_.quality.low_utility
        overview = situation.get("market_overview", [])
        self.varies = {r["product_id"] for r in overview if r.get("quality_varies")}
        self.asked: set[tuple[int, str, str]] = set()

    def _expected(self, pid: str, next_utility: float, seller_rating: Any = None,
                  certified: bool = False) -> float:
        if pid not in self.varies or certified:
            return next_utility
        p_low = 1.0 if seller_rating and seller_rating["average"] < self.avoid_rating else self.p_low
        return next_utility * (1 - p_low * (1 - self.low_utility))

    def decide(self, obs: dict[str, Any]) -> Decision:
        if self.review:
            for t in obs.get("your_unreviewed_trades", []):
                low = t.get("grade") == "low"
                return Decision("post_review", {
                    "trade_id": t["trade_id"], "rating": 1 if low else 5,
                    "text": "Received low grade." if low else "As described."})
        nxt = obs["you"]["utility_of_next_unit"]
        cash = obs["you"]["cash"]
        late = obs["round"] >= self.panic_round
        ratings = {l["seller"]: l.get("seller_rating") for l in obs.get("board", [])}
        good = []
        for o in obs.get("offers_to_you", []):
            pid = o["product_id"]
            if pid not in nxt or o["qty"] != 1 or o["price"] > cash:
                continue
            v = self._expected(pid, nxt[pid], ratings.get(o["seller"]), o.get("certified_high", False))
            if o["price"] <= v * (1.0 if late else self.accept):
                good.append((v - o["price"], o))
        if good:
            return Decision("accept_offer", {"offer_id": max(good, key=lambda g: g[0])[1]["offer_id"]})
        options = []
        for l in obs.get("board", []):
            pid = l["product_id"]
            if pid in nxt and l["price"] <= cash:
                v = self._expected(pid, nxt[pid], l.get("seller_rating"), l.get("certified_high", False))
                options.append((v - l["price"], v, l))
        options.sort(key=lambda o: o[0], reverse=True)
        for gain, v, l in options:
            if l["price"] <= v * self.target or (late and gain > 0):
                return Decision("buy_listed", {"seller_id": l["seller"],
                                               "product_id": l["product_id"], "qty": 1})
        for gain, v, l in options:
            key = (obs["round"], l["seller"], l["product_id"])
            if key not in self.asked and v > 0:
                self.asked.add(key)
                ask = round(v * self.target, 2)
                return Decision("send_message", {
                    "to": l["seller"],
                    "message": f"Could you do ${ask:.2f} for 1 unit of {l['product_id']}?"})
        return Decision("wait", {})


class HeuristicSeller(Policy):
    """Simple baseline sellers, honest (always high grade), with or without a public board.

    kind "zi"     zero intelligence, budget-constrained (Gode & Sunder 1993): each turn posts
                  or offers a random product it still has at a price drawn uniformly from
                  [unit cost, params.max_factor (2.0) x list price]: random, but never at a loss.
    kind "markup" always prices at params.markup (1.3) x unit cost.
    kind "list"   always prices at the list price (at least cost).
    With a board the price goes on the board (re-drawn every turn for zi); without one the
    seller makes a binding offer of 1 unit to a random buyer.
    """

    def __init__(self, agent: AgentSpec, situation: dict[str, Any], spec: Any, kind: str):
        p = agent.params
        self.kind = kind
        self.rng = random.Random(f"{kind}:{agent.id}:{situation.get('your_id')}:"
                                 f"{','.join(x['product_id'] for x in situation['products'])}")
        self.cost = situation["your_unit_cost"]
        self.list_price = {x["product_id"]: x["list_price"] for x in situation["products"]}
        enforced = situation.get("your_minimum_price", {})
        self.floor = {pid: max(c, enforced.get(pid, 0)) for pid, c in self.cost.items()}
        self.buyers = list(situation.get("buyers", []))
        self.board = spec.channels.board
        self.markup = p.get("markup", 1.3)
        self.max_factor = p.get("max_factor", 2.0)
        self.posted: dict[str, float] = {}

    def _price(self, pid: str) -> float:
        lo = self.floor[pid]
        if self.kind == "zi":
            hi = max(lo, self.list_price[pid] * self.max_factor)
            return round(self.rng.uniform(lo, hi), 2)
        target = self.cost[pid] * self.markup if self.kind == "markup" else self.list_price[pid]
        return round(max(lo, target), 2)

    def decide(self, obs: dict[str, Any]) -> Decision:
        stock = [pid for pid, q in obs["you"]["stock"].items() if q > 0 and pid in self.cost]
        if not stock:
            return Decision("wait", {})
        if self.board:
            todo = stock if self.kind == "zi" else [p for p in stock if p not in self.posted]
            if not todo:
                return Decision("wait", {})
            pid = self.rng.choice(todo)
            self.posted[pid] = price = self._price(pid)
            return Decision("set_price", {"product_id": pid, "price": price})
        if not self.buyers:
            return Decision("wait", {})
        pid = self.rng.choice(stock)
        return Decision("make_offer", {"buyer_id": self.rng.choice(self.buyers), "product_id": pid,
                                       "price": self._price(pid), "qty": 1})


class HeuristicBuyer(Policy):
    """Simple baseline buyers, with or without a public board. Never pay more than the
    next unit's utility, nor more than the cash left.

    kind "zi"     zero intelligence, budget-constrained: takes a random acceptable deal (board
                  listing or offer to it); without a board and without offers it asks a random
                  seller of a product it still values for an offer.
    kind "greedy" truthful and impatient: takes the deal with the largest gain right away.
    """

    def __init__(self, agent: AgentSpec, situation: dict[str, Any], spec: Any, kind: str):
        self.kind = kind
        self.rng = random.Random(f"{kind}:{agent.id}:{situation.get('your_id')}")
        self.sellers = list(situation.get("sellers", []))

    def decide(self, obs: dict[str, Any]) -> Decision:
        nxt, cash = obs["you"]["utility_of_next_unit"], obs["you"]["cash"]
        deals = []
        for o in obs.get("offers_to_you", []):
            if o["product_id"] in nxt and o["price"] * o["qty"] <= cash and o["qty"] == 1 \
                    and o["price"] <= nxt[o["product_id"]]:
                deals.append((nxt[o["product_id"]] - o["price"], "accept_offer", {"offer_id": o["offer_id"]}))
        for l in obs.get("board", []):
            if l["product_id"] in nxt and l["price"] <= cash and l["price"] <= nxt[l["product_id"]]:
                deals.append((nxt[l["product_id"]] - l["price"], "buy_listed",
                              {"seller_id": l["seller"], "product_id": l["product_id"], "qty": 1}))
        if deals:
            gain, tool, args = (max(deals, key=lambda d: d[0]) if self.kind == "greedy"
                                else self.rng.choice(deals))
            return Decision(tool, args)
        if "board" not in obs and nxt and self.sellers and self.rng.random() < 0.5:
            pid = self.rng.choice(sorted(nxt))
            return Decision("send_message", {"to": self.rng.choice(self.sellers),
                                             "message": f"Please send me an offer for 1 unit of {pid}."})
        return Decision("wait", {})


SCRIPTED: dict[str, Callable[..., Policy]] = {
    "haggling_seller": lambda agent, sit, spec: ScriptedSeller(agent, sit, spec),
    "bargain_buyer": lambda agent, sit, spec: ScriptedBuyer(agent, sit, spec),
    "zi_seller": lambda agent, sit, spec: HeuristicSeller(agent, sit, spec, "zi"),
    "markup_seller": lambda agent, sit, spec: HeuristicSeller(agent, sit, spec, "markup"),
    "list_seller": lambda agent, sit, spec: HeuristicSeller(agent, sit, spec, "list"),
    "zi_buyer": lambda agent, sit, spec: HeuristicBuyer(agent, sit, spec, "zi"),
    "greedy_buyer": lambda agent, sit, spec: HeuristicBuyer(agent, sit, spec, "greedy"),
}

MOCK_STRATEGY = {"seller": "haggling_seller", "buyer": "bargain_buyer"}
