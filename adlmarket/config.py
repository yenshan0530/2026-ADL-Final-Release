"""Run configuration: which models exist, which agents use them, and who sits where.

Three layers, each its own YAML so they can be swapped independently:

  models   -- a registry of named LLM endpoints (base model or LoRA adapter on a vLLM server,
              or a hosted API). Agents refer to models by name, never by URL.
  agents   -- one file per agent: role, controller (llm | scripted), model name, persona.
              TA personas and student submissions use the exact same format.
  run      -- the market rules plus *casting*: which agent pools fill the seller and buyer
              seats, how many per market, and how the graded side is partitioned.

Stage 1, 2 and 3 are not different games, only different casting:
  stage 1: sellers <- student submissions, buyers <- TA personas
  stage 2: sellers <- TA personas,         buyers <- student submissions
  stage 3: both sides <- student submissions
"""

from __future__ import annotations

import glob
import math
import re
import os
import re
import zlib
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

Role = Literal["seller", "buyer"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------- models ---

class LoraSpec(_Strict):
    """A LoRA adapter served on top of `base_model` (vLLM `--lora-modules name=path`)."""

    base_model: str
    path: str


class ModelSpec(_Strict):
    """One callable model behind an OpenAI-compatible endpoint: a local vLLM server (base
    model or LoRA) or a hosted API (OpenAI, Anthropic, Gemini, NVIDIA, ...). `model` is the
    name the endpoint knows it by; for a LoRA it is the adapter name."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: Optional[str] = None      # env var (or .env entry) holding the key: preferred
    api_key: Optional[str] = Field(default=None, repr=False)   # inline key; never logged
    extra_body: dict[str, Any] = Field(default_factory=dict)   # provider-specific request fields
    lora: Optional[LoraSpec] = None
    # Rename LoRA weights before the server loads them: {old prefix: new prefix}. vLLM serves
    # Qwen3.5 as its multimodal class, whose layers are model.language_model.layers.*, but
    # PEFT on the text model saves model.layers.*: unrenamed, vLLM accepts the adapter and
    # silently applies none of it. The student's files are not changed (a renamed copy is
    # loaded from the run folder).
    lora_rename: dict[str, str] = Field(default_factory=dict)
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    # Reasoning models spend part of this on thinking; 1024 cut their calls off (ADL-26W).
    max_tokens: int = 8192
    # Turns whose reply hits max_tokens are asked again this many times (with a nudge to
    # think less) before the turn is lost as `no_tool_call`.
    length_retries: int = Field(default=1, ge=0)
    temperature: float = 0.7
    top_p: float = 0.95
    # Several servers holding this same model (one per GPU or GPU group). Filled from the
    # `serving` section, or list them yourself for servers you started. Empty: base_url.
    replicas: list[str] = Field(default_factory=list)

    def url_for(self, key: str) -> str:
        """The server for one agent or adapter: a stable hash of `key` over the replicas,
        so a LoRA always lands on the same server and stays in its adapter cache."""
        if not self.replicas:
            return self.base_url
        return self.replicas[zlib.crc32(key.encode()) % len(self.replicas)]

    def resolve_key(self) -> str:
        """The key to send: inline `api_key`, else `api_key_env` from the environment or a
        local .env file, else "EMPTY" (local vLLM accepts anything)."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            from adlmarket.client import _env_or_dotenv
            key = _env_or_dotenv(self.api_key_env)
            if not key:
                raise RuntimeError(f"model {self.model!r}: {self.api_key_env} is not set "
                                   f"(export it or add it to .env)")
            return key
        return "EMPTY"

    def public_dump(self) -> dict[str, Any]:
        """For logs: everything except the key itself."""
        out = self.model_dump(exclude={"api_key"})
        out["api_key"] = "<inline>" if self.api_key else None
        return out


class ServingSpec(_Strict):
    """How to run a registry model on local GPUs with vLLM: one server ("replica") per
    entry of `gpus`, each holding the base model once. LoRA adapters are registered at
    runtime and shared: vLLM keeps up to `max_loras` on the GPU, mixes requests for
    different adapters in one batch, and swaps from a CPU cache of `max_cpu_loras`.

    gpus: [0, 1, 2]   three replicas, one GPU each
    gpus: ["0,1"]     one replica with tensor parallelism over GPUs 0 and 1
    """

    gpus: list[int | str] = Field(default_factory=lambda: [0])
    host: str = "localhost"
    port: int = 8000                    # replica i listens on port + i
    max_loras: int = 8
    max_cpu_loras: int = 64
    max_lora_rank: int = 64
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 64000          # prompt (system + history, up to ~20k tokens) + reply (max_tokens)
    tool_call_parser: str = "hermes"
    command: list[str] = Field(default_factory=lambda: ["vllm", "serve"])
    extra_args: list[str] = Field(default_factory=list)
    startup_timeout: float = 900.0

    def urls(self) -> list[str]:
        return [f"http://{self.host}:{self.port + i}/v1" for i in range(len(self.gpus))]


# ------------------------------------------------------------------------- agents ---

GUARDS: dict[str, tuple[str, ...]] = {"seller": ("min_markup",),
                                      "buyer": ("max_price_to_utility",)}


class AgentSpec(_Strict):
    """One agent: a role, how it decides (LLM or scripted rules), and its persona."""

    id: str
    role: Role
    controller: Literal["llm", "scripted"] = "llm"
    owner: str = "ta"                      # "ta" or a student id; scores are reported per owner
    # A name from the model registry, or a whole model inline, e.g. a hosted API only this
    # agent uses: {model: gpt-5.5, base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY}
    model: Optional[str | ModelSpec] = None
    # Path to a LoRA adapter folder, relative to the agent file. The agent then runs `model`
    # with it on top; the harness registers it as model `lora-<id>` on the same server.
    lora: Optional[str] = None
    persona: Optional[str] = None          # path to a text file, relative to the agent file
    persona_text: Optional[str] = None     # or the persona inline
    # Several personas: files or globs relative to the agent file. The agent is then played
    # once per persona (same model / LoRA / owner), as agents `<id>_<persona file stem>`.
    personas: Optional[list[str]] = None
    # Thinking mode for models that have one (Qwen3.5: chat_template_kwargs.enable_thinking).
    # Off unless the agent file says `reasoning: true`. Must match how the LoRA was trained
    # (scripts/train_lora.py --thinking / --no-thinking).
    reasoning: bool = False
    variant_of: Optional[str] = None       # set on those copies: the id of the agent file
    personas_given: Optional[int] = None   # on those copies: personas the student handed in
    strategy: Optional[str] = None         # scripted only: name in adlmarket.policies.SCRIPTED
    params: dict[str, Any] = Field(default_factory=dict)
    # Engine-enforced limits the LLM cannot talk its way past (see GUARDS):
    #   seller {"min_markup": 1.1}: never price below 1.1 x unit cost
    #   buyer  {"max_price_to_utility": 1.0}: never pay more than the unit is worth
    guards: dict[str, float] = Field(default_factory=dict)
    source: Optional[str] = None           # file this spec was loaded from (filled in by loader)

    @model_validator(mode="after")
    def _check_controller(self) -> "AgentSpec":
        if self.controller == "llm" and not self.model:
            raise ValueError(f"agent {self.id}: an llm agent needs `model`")
        if self.controller == "scripted" and not self.strategy:
            raise ValueError(f"agent {self.id}: a scripted agent needs `strategy`")
        for name, value in self.guards.items():
            if name not in GUARDS[self.role]:
                raise ValueError(f"agent {self.id}: unknown {self.role} guard {name!r}; "
                                 f"known: {', '.join(GUARDS[self.role])}")
            if not value > 0:
                raise ValueError(f"agent {self.id}: guard {name} must be > 0, got {value}")
        return self

    def persona_body(self) -> str:
        if self.persona_text:
            return self.persona_text.strip()
        if self.persona:
            return self.persona_path().read_text(encoding="utf-8").strip()
        return ""

    def persona_path(self) -> Path:
        """The persona file; it must sit inside the agent's own folder."""
        base = (Path(self.source).parent if self.source else Path(".")).resolve()
        path = (base / self.persona).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"agent {self.id}: persona {self.persona!r} is outside {base}")
        return path


# ------------------------------------------------------------------------- market ---

class RoundsSpec(_Strict):
    """Market length: every market lasts exactly `total` rounds of `turns_per_round` turns,
    and agents are told `total`.

    free_messages: per turn, up to this many successful messages (send_message /
    post_to_room) do not use the turn: the agent acts again in the same turn. 0 = off.

    The random length (min, max, continue_prob) was removed. Old configs with a fixed
    length (min == max, or continue_prob 0) still load as `total`; others are refused."""

    total: int = Field(default=5, ge=1)
    turns_per_round: int = Field(default=6, ge=1)
    free_messages: int = Field(default=0, ge=0)
    # Output tokens (visible + thinking) each LLM agent may generate per round, over all its
    # calls. A call may use at most what is left (a longer reply is cut off there, as at
    # max_tokens); with nothing left the agent waits until the next round. None: no limit.
    output_tokens_per_round: Optional[int] = Field(default=8192, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _legacy_length(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not {"min", "max", "continue_prob"} & set(data):
            return data
        data = dict(data)
        lo, hi = data.pop("min", None), data.pop("max", None)
        p = data.pop("continue_prob", None)
        fixed = lo if (hi is None or hi == lo or p == 0) else None
        if fixed is None or "total" in data:
            raise ValueError("rounds.min / max / continue_prob were removed: markets last a "
                             "fixed number of rounds, `rounds.total` (default 5)")
        data["total"] = fixed
        return data


class KindSpec(_Strict):
    """One kind of item. Which items have it is set in the item file (each item's `kind`).

    supply_ratio: units for sale / buyers in the market (each buyer wants one unit;
                  > 1: sellers compete; < 1: buyers compete for scarce stock).
    sellers:      "all" sellers stock it, "one" (a monopoly), or how many do (3 = an oligopoly of three).
                  A market with no more sellers than that gives it to all of them. Items
                  sold by some sellers are dealt to the seller seats in item-file order, as
                  evenly as possible: 5 monopoly items give 5 sellers one each, and 3
                  sellers 2, 2 and 1 (rotation evens that out).
    max_share:    with a number of `sellers`: at most this share of the market's sellers,
                  rounded up, hold each item. Oligopoly {sellers: 3, max_share: 0.5}: 3 of 5
                  sellers, 2 of 3, 1 of 2.
    """

    supply_ratio: float
    sellers: Union[Literal["all", "one"], PositiveInt] = "all"
    max_share: Optional[float] = Field(default=None, gt=0, le=1)

    @property
    def holders(self) -> Optional[int]:
        """How many sellers stock it, before max_share; None means all of them."""
        return {"all": None, "one": 1}.get(self.sellers, self.sellers)

    def holders_in(self, n_sellers: int) -> Optional[int]:
        """How many of a market's `n_sellers` sellers stock each item; None means all."""
        h = self.holders
        if h is not None and self.max_share is not None:
            h = min(h, max(1, math.ceil(self.max_share * n_sellers)))
        return h


def _default_kinds() -> dict[str, KindSpec]:
    """The kinds of the graded economy (configs/economy.yaml)."""
    return {
        "monopoly": KindSpec(sellers="one", supply_ratio=1.0),
        "oligopoly": KindSpec(sellers=3, max_share=0.5, supply_ratio=1.0),
        "scarce": KindSpec(supply_ratio=0.6),
        "competitive": KindSpec(supply_ratio=1.5),
        "balanced": KindSpec(supply_ratio=1.0),
    }


class QualitySpec(_Strict):
    """Items a seller can deliver in a high or a low grade (`low_grade: true` in the item
    file). The seller chooses the grade (hidden from the buyer) when listing or offering;
    the buyer learns it on delivery.

    low_utility: a low-grade unit's utility, x the high-grade utility of that unit.
    low_cost:    a low-grade unit's cost to the seller, x the high-grade cost.
    certification: sellers may mark a listing/offer "certified high"; the engine only
                   allows it on high-grade units, so it is a credible signal.
    reviews:     buyers may review a purchase (post_review); reviews are public.
    disclosure:  "private": only the buyer learns the grade; "public": every agent is told.
    """

    low_utility: float = 0.4
    low_cost: float = 0.5
    certification: bool = False
    reviews: bool = True
    disclosure: Literal["private", "public"] = "private"


class SchemaSpec(_Strict):
    """What one economy (or one run_market repeat) is played over: its items, their kinds
    and prices, all from the item file; nothing is drawn at random. Every market of the
    economy plays the same items, so the whole class faces the same problem.

    Every buyer values one unit of each item at its buyer value (more units are worth nothing,
    but may be bought); an item's
    units for sale = its kind's supply_ratio x buyers in the market.
    """

    kinds: dict[str, KindSpec] = Field(default_factory=_default_kinds)
    quality: QualitySpec = QualitySpec()
    # Item file (required): every item's seller cost and buyer value; the economy plays the
    # items that have a `kind` (low_grade: true for items that can be delivered in a low
    # grade). One file for every economy, or a list with one file per economy, exactly as
    # many as `repeats` (e.g. [items_1.json, items_2.json, items_3.json] for 3 economies).
    items: Union[str, list[str], None] = None
    # Each buyer's cash = (1 + budget_extra) x sum over all units for sale of
    # mean(seller cost, buyer value), / number of buyers in the market.
    budget_extra: float = 0.15

    @field_validator("items", mode="before")
    @classmethod
    def _one_file(cls, v: Any) -> Any:
        return v[0] if isinstance(v, list) and len(v) == 1 else v

    def items_file(self, economy: int) -> Optional[str]:
        """The item file of economy `economy`."""
        return self.items[economy] if isinstance(self.items, list) else self.items


class ChannelSpec(_Strict):
    board: bool = True          # public price list
    dm: bool = True             # private 1-to-1 messages
    dm_same_side: bool = True   # may a seller DM a seller / a buyer DM a buyer
    seller_room: bool = True    # group chat for the sellers of a market
    buyer_room: bool = True     # group chat for the buyers of a market


class ScoringSpec(_Strict):
    unsold_salvage: float = 0.0     # value of an unsold unit, x its cost (0 = worthless)
    # Buyers may buy units beyond their utility table (worth nothing to them: money lost).
    # On: the engine does not stop irrational trades; false rejects them (exceeds_need).
    allow_excess_purchase: bool = True


class MarketSpec(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    catalog: str = "scenarios/public/catalog.yaml"
    rounds: RoundsSpec = RoundsSpec()
    schema_: SchemaSpec = Field(default_factory=SchemaSpec, alias="schema")
    channels: ChannelSpec = ChannelSpec()
    scoring: ScoringSpec = ScoringSpec()
    offer_ttl_rounds: int = 2           # a binding offer stays open this round and the next
    max_open_offers: int = 8            # per seller
    max_message_chars: int = 1000       # longest message an agent may send; longer: rejected (message_too_long)
    private_state_chars: int = 1000
    persona_chars: int = 8000           # longer personas are rejected before the run starts
    history_turns: int = 6              # past turns an LLM agent sees verbatim
    # Board in past turns: "full" keeps each past observation as it was; "latest" drops the
    # board from past observations (only the current turn shows it), which keeps prompts
    # from growing with sellers x products.
    history_board: Literal["full", "latest"] = "full"
    # Text about each product in the system prompt: "summary" (the catalog's one-line
    # summary), "description" (the catalog description, cut to description_chars) or "none".
    product_text: Literal["summary", "description", "none"] = "summary"
    # With product_text "description": first N characters (280, as in the TA engine), None
    # for the full text.
    description_chars: Optional[int] = 280
    # Totals per product (units for sale, sellers stocking it, units buyers want at list
    # price, whether quality varies), shown to everyone: "start" in the system prompt,
    # "live" also as remaining stock each turn, "none" hidden.
    market_overview: Literal["start", "live", "none"] = "start"
    # Buyer utilities: "shared" = every buyer has the schema's table; "private" = each buyer's
    # table for each product is the schema's table x a factor drawn uniformly from
    # [1 - spread, 1 + spread] (same draw for every unit of that product), known only to it.
    buyer_values: Literal["shared", "private"] = "shared"
    # Buyers may trade units they hold and cash with each other: propose_swap (binding until
    # it expires) / accept_swap. Pair with scoring.allow_excess_purchase to let buyers buy
    # for others (e.g. one buys two goods in bulk and swaps).
    buyer_swaps: bool = False
    private_value_spread: float = Field(default=0.4, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _swaps_need_dm(self) -> "MarketSpec":
        if self.buyer_swaps and not (self.channels.dm and self.channels.dm_same_side):
            raise ValueError("buyer_swaps needs channels.dm and channels.dm_same_side: a swap "
                             "is proposed by a buyer-to-buyer DM")
        return self


# ------------------------------------------------------------------------ casting ---

class SideCasting(_Strict):
    """Who fills one side's seats.

    mode:
      partition -- the graded side: every agent plays in exactly one market per repeat.
      sample    -- the opponent side: each market draws `per_market` agents from the pool,
                   balanced so every agent is used about equally.
      all       -- every agent of the pool sits in every market (small pools only).
    """

    agents: list[str | dict[str, Any]]  # agent YAML paths or globs, or {agent, personas, lora}
    per_market: int = 3
    mode: Literal["partition", "sample", "all"] = "sample"
    # partition only: every market has exactly `per_market` agents of this side; markets =
    # ceil(agents / per_market), and markets short of agents are topped up with agents
    # drawn from `pad` (e.g. TA sellers; balanced, copies when needed). Not graded.
    # Without `exact`, `pad` only tops up markets with fewer than `per_market` agents.
    exact: bool = False
    pad: list[str | dict[str, Any]] = Field(default_factory=list)


class Casting(_Strict):
    sellers: SideCasting
    buyers: SideCasting
    # How many times to play: each repeat re-partitions the partitioned side into fresh
    # markets (an "economy"). Without a partitioned side, each repeat is one market.
    repeats: int = 1
    # Markets per repeat: "min" = per_market is the smallest market size (floor, fewest
    # over the partitioned sides); "max" = per_market is the largest (ceil, most over the
    # sides; the rule before 2026-11-08). See adlmarket.casting.
    market_count: Literal["min", "max"] = "min"
    # Play every market once per seller seat, shifting which seat holds which stock
    # (bundle) by one each time, so every seller holds every bundle once: market e0_m1
    # becomes e0_m1_r0 ... e0_m1_r4 with 5 sellers. Same agents, schema and random seed.
    rotate: bool = False


class GradingSpec(_Strict):
    """Grades from TA anchors (phases 1 and 2). Per economy, the TA agents of the graded
    role are ranked by mean raw score (seller profit, buyer surplus); the TA at
    `low_percentile` gets `low_score` and the TA at `high_percentile` gets `high_score`
    (position floor(p/100 x (n - 1)) from the lowest: with few TAs, the lower-scoring one,
    which favors students). A student's grade is the straight line through the two
    anchors at its raw score, not capped; its final grade is the mean over economies."""

    low_percentile: float = Field(default=10, ge=0, le=100)
    low_score: float = 0.15
    high_percentile: float = Field(default=80, ge=0, le=100)
    high_score: float = 0.85

    @model_validator(mode="after")
    def _order(self) -> "GradingSpec":
        if self.low_percentile >= self.high_percentile:
            raise ValueError("grading: low_percentile must be below high_percentile")
        return self


class RunConfig(_Strict):
    name: str
    seed: int = 0
    models: list[str] = Field(default_factory=lambda: ["configs/models.yaml"])
    market: MarketSpec = MarketSpec()
    casting: Casting
    # Personas each student agent plays (see persona_variants): its LoRA with personas 1..N,
    # padded with persona 1. None: every persona listed in `personas`, unpadded.
    personas_per_student: Optional[int] = Field(default=None, ge=1)
    output_dir: str = "results"     # runs go to <output_dir>/<script>_<timestamp>/
    max_workers: int = 8                # concurrent LLM calls within one market tick
    # Markets played at the same time. More keeps a shared GPU busy (its batch mixes calls
    # from every running market); concurrent calls = parallel_markets x max_workers.
    parallel_markets: int = 1
    # Save every turn in full (observation, tool call, result, model output) and system
    # prompts, as markets/<id>/events.jsonl. Off or on, every action (transcript.jsonl /
    # .md), all messages and trades (final.json), scores and standings are written.
    save_trajectories: bool = True
    # Also save the exact messages each LLM call was sent (needs save_trajectories). Large;
    # meant for building training data.
    save_prompts: bool = False
    min_call_interval: float = 0.0      # seconds between calls to one model (hosted APIs
                                        # with tight rate limits); 0 for local vLLM
    call_timeout: float = 120.0         # seconds per model call attempt
    call_retries: int = 4               # retries after a server error / timeout / rate limit
    # Stop the run when one agent's model calls fail (after retries) this many turns in a
    # row: its endpoint is down, and scoring the market would be meaningless.
    max_failed_calls_in_a_row: int = 8
    # Roles whose students are graded, and how: grading from TA anchors, or None (ranking).
    graded_roles: list[str] = Field(default_factory=list)
    grading: Optional[GradingSpec] = None
    source: Optional[str] = None
    # The run file and how it was run (script, phase); the file is saved as
    # <output>/config.yaml, and --resume reads it back.
    origin: dict[str, Any] = Field(default_factory=dict)

    @field_validator("models", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    def check_items(self) -> None:
        """The item file(s): required; one, or exactly one per economy (repeats)."""
        spec, repeats = self.market.schema_, self.casting.repeats
        if not spec.items:
            raise ValueError("market.schema.items is required: the item file each economy "
                             "plays (e.g. scenarios/public/items_example.json)")
        if isinstance(spec.items, list) and len(spec.items) != repeats:
            raise ValueError(f"market.schema.items lists {len(spec.items)} files but repeats is "
                             f"{repeats}: give one file (used by every economy) or exactly one "
                             f"per economy")


# ------------------------------------------------------------------------ loading ---

def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_models(paths: list[str], root: Path) -> tuple[dict[str, ModelSpec], dict[str, ServingSpec]]:
    """The model registry and the local serving setup, from one or more models files."""
    registry: dict[str, ModelSpec] = {}
    serving: dict[str, ServingSpec] = {}
    for p in paths:
        data = _read_yaml(root / p)
        for name, spec in (data.get("models") or {}).items():
            if name in registry:
                raise ValueError(f"model {name!r} defined twice")
            registry[name] = ModelSpec(**spec)
        for name, spec in (data.get("serving") or {}).items():
            serving[name] = ServingSpec(**spec)
    for name, sv in serving.items():
        if name not in registry:
            raise ValueError(f"serving: unknown model {name!r}")
        registry[name].replicas = sv.urls()
        registry[name].base_url = sv.urls()[0]
    # Models naming the same weights at the same address share those servers.
    for m in registry.values():
        for name, sv in serving.items():
            if m.model == registry[name].model and not m.replicas and \
                    m.base_url in (registry[name].base_url, *sv.urls()):
                m.replicas, m.base_url = list(registry[name].replicas), registry[name].base_url
    return registry, serving


ROLES = ("seller", "buyer")
PERSONA_FILE = "persona.md"
PERSONAS_GLOB = "persona_*.md"
CHECKPOINT_DIR = "checkpoint"
MAX_LORA_RANK = 32            # students' LoRAs; the server (max_lora_rank) also loads TA ones
TA_MAX_LORA_RANK = 64         # the TA agents' LoRAs (agents/_ta)


# Model settings an agent file may give at its top level, next to `model: <served name>`,
# instead of naming a models.yaml entry. `endpoint` is the server's base URL.
FLAT_MODEL_FIELDS = {"endpoint": "base_url", "api_key": "api_key", "api_key_env": "api_key_env",
                     "max_tokens": "max_tokens", "temperature": "temperature", "top_p": "top_p",
                     "reasoning_effort": "reasoning_effort", "extra_body": "extra_body",
                     "length_retries": "length_retries"}


def _fold_flat_model(data: dict[str, Any], path: Path) -> None:
    """`model: gpt-6.1-terra` + `endpoint: https://...` + `api_key: ...` + `max_tokens: ...`
    at the top of an agent file -> one inline model spec."""
    flat = {k: data.pop(k) for k in list(data) if k in FLAT_MODEL_FIELDS}
    if not flat:
        return
    if "endpoint" not in flat:
        raise ValueError(f"{path}: model settings ({', '.join(flat)}) need `endpoint`, the "
                         f"server's base URL")
    if not isinstance(data.get("model"), str):
        raise ValueError(f"{path}: with `endpoint`, `model` must be the served model name")
    data["model"] = {"model": data["model"],
                     **{FLAT_MODEL_FIELDS[k]: v for k, v in flat.items()}}


def is_agent_folder(folder: Path) -> bool:
    """A submission role folder (`<student>/seller/`, `<student>/buyer/`): it has an
    agent.yaml, a checkpoint/ or a persona file. agent.yaml is optional."""
    return folder.name in ROLES and folder.is_dir() and (
        (folder / "agent.yaml").is_file() or (folder / CHECKPOINT_DIR).is_dir()
        or any(folder.glob("persona*.md")))


def load_agent(path: Path) -> AgentSpec:
    """Load one agent file.

    A file at `agents/<student_id>/<role>/agent.yaml` (or `<student_id>/agents/<role>/`, the
    layout of a submission) gets its defaults from the folder:

        agents/<student_id>/<role>/
            agent.yaml       may be empty; any field here overrides the defaults
            persona_1.md ... the personas (persona_*.md): the agent plays once with each
            persona.md       or a single persona (used when there is no persona_*.md)
            checkpoint/      LoRA adapter from PEFT save_pretrained (optional)

    defaults: id <student_id>_<role>, role and owner from the path, controller llm, model
    `base`, personas persona_*.md (else persona persona.md) and lora checkpoint/ when
    they exist.
    """
    folder = path.parent
    if path.name == "agent.yaml" and not path.exists() and is_agent_folder(folder):
        data: dict[str, Any] = {}                # no agent.yaml: all defaults from the folder
    else:
        data = _read_yaml(path)
    _fold_flat_model(data, path)
    if path.name == "agent.yaml" and folder.name in ROLES:
        # agents/<student_id>/<role>/ (the repository), or <student_id>/agents/<role>/ (a
        # submission, docs/SUBMISSION.md): the student id is the folder above, or above agents/
        owner = folder.parent.parent.name if folder.parent.name == "agents" else folder.parent.name
        data.setdefault("role", folder.name)
        data.setdefault("owner", owner)
        data.setdefault("id", f"{owner}_{folder.name}")
        if data.setdefault("controller", "llm") == "llm":
            data.setdefault("model", "base")
        if not {"persona", "persona_text", "personas"} & set(data):
            if sorted(folder.glob(PERSONAS_GLOB)):
                data["personas"] = [PERSONAS_GLOB]
            elif (folder / PERSONA_FILE).exists():
                data["persona"] = PERSONA_FILE
        if "lora" not in data and (folder / CHECKPOINT_DIR / "adapter_config.json").exists():
            data["lora"] = CHECKPOINT_DIR
    data["source"] = str(path)
    return AgentSpec(**data)


def check_credentials(run: "LoadedRun") -> None:
    """Fail before playing if any LLM agent's model has no usable key."""
    for a in run.everyone():
        if a.controller == "llm":
            run.models[a.model].resolve_key()


def check_adapter(agent_id: str, path: Path, base_model: str,
                  rename: Optional[dict[str, str]] = None, max_rank: int = MAX_LORA_RANK) -> None:
    """Reject adapters the grading server could not load on the course base model."""
    cfg_file = path / "adapter_config.json"
    if not cfg_file.exists():
        raise ValueError(f"agent {agent_id}: {path} has no adapter_config.json "
                         f"(save it with PEFT `save_pretrained`)")
    if not any((path / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
        hint = (" (TA LoRA weights: python scripts/download_ta_loras.py)"
                if "/_ta/" in str(path) else "")
        raise ValueError(f"agent {agent_id}: {path} has no adapter_model.safetensors{hint}")
    import json
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"agent {agent_id}: adapter must be LoRA, got {cfg.get('peft_type')}")
    if int(cfg.get("r", 0)) > max_rank:
        raise ValueError(f"agent {agent_id}: LoRA rank {cfg.get('r')} > max {max_rank}")
    pattern_rank = max((int(v) for v in (cfg.get("rank_pattern") or {}).values()), default=0)
    tensor_rank = _adapter_rank(_adapter_header(path / "adapter_model.safetensors"))
    if max(pattern_rank, tensor_rank) > max_rank:
        raise ValueError(f"agent {agent_id}: LoRA has layers of rank {max(pattern_rank, tensor_rank)} "
                         f"(rank_pattern / weight shapes) > max {max_rank}")
    trained_on = cfg.get("base_model_name_or_path")
    if trained_on and trained_on != base_model and Path(trained_on).name != Path(base_model).name:
        raise ValueError(f"agent {agent_id}: adapter was trained on {trained_on}, "
                         f"but this agent runs on {base_model}")
    keys = _adapter_keys(path / "adapter_model.safetensors")
    if any(".visual." in k or "embed_tokens" in k or "lm_head" in k for k in keys):
        raise ValueError(f"agent {agent_id}: the LoRA may only adapt the language model's "
                         f"layers, not the vision encoder, embeddings or lm_head")
    # Qwen3.5 checkpoints keep the text layers under `model.language_model.layers`. PEFT on
    # AutoModelForCausalLM saves them as `model.layers`, and vLLM loads such an adapter
    # without error but never applies it.
    # With the model's `lora_rename` mapping model.layers. the harness loads a renamed copy.
    renamed = any("model.layers." in old for old in (rename or {}))
    if (not renamed and "qwen3.5" in base_model.lower()
            and any(re.search(r"\.model\.layers\.\d", k) for k in keys)):
        raise ValueError(f"agent {agent_id}: adapter keys use `model.layers.N`, which vLLM "
                         f"silently ignores on {base_model}; set the model's `lora_rename: "
                         f"{{\"model.layers.\": \"model.language_model.layers.\"}}` in "
                         f"configs/models.yaml (the harness then loads a renamed copy)")


def _adapter_header(path: Path) -> dict[str, Any]:
    """Tensor names and shapes in a safetensors file, from its header (no torch needed)."""
    import json
    import struct
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            return {k: v for k, v in json.loads(f.read(size)).items() if k != "__metadata__"}
    except (OSError, struct.error, ValueError):
        return {}


def _adapter_keys(path: Path) -> list[str]:
    return list(_adapter_header(path))


def _adapter_rank(header: dict[str, Any]) -> int:
    """The largest LoRA rank in the weights: lora_A is [r, in], lora_B is [out, r]."""
    ranks = [v["shape"][0] for k, v in header.items() if "lora_A" in k and v.get("shape")]
    ranks += [v["shape"][-1] for k, v in header.items() if "lora_B" in k and v.get("shape")]
    return max(ranks, default=0)


def _natural(name: str) -> list:
    """persona_2.md before persona_10.md."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _persona_list(agent: AgentSpec) -> list[str]:
    """The agent's persona files (globs expanded, in natural order), or [] if none."""
    base = Path(agent.source).parent if agent.source else Path(".")
    files: list[str] = []
    for pat in agent.personas or []:
        hits = (sorted((Path(m).relative_to(base).as_posix() for m in glob.glob(str(base / pat))),
                       key=_natural)
                if any(c in pat for c in "*?[") else [pat] if (base / pat).is_file() else [])
        if not hits and not any(c in pat for c in "*?["):
            raise ValueError(f"agent {agent.id}: persona {pat!r} is not a file in {base}")
        files += [h for h in hits if h not in files]
    return files


def persona_variants(agent: AgentSpec, count: Optional[int] = None) -> list[AgentSpec]:
    """One agent per persona.

    count None (plain run configs): an agent with `personas` plays each of them, as
    `<id>_<persona file stem>`; any other agent plays as itself.
    count N (personas_per_student): every student agent (owner other than "ta") plays
    exactly N copies `<id>_p1` ... `<id>_pN` with its personas 1..N in order; with fewer
    than N personas the rest are persona 1 again (2 given, N = 3: 1, 2, 1), and with none
    every copy has an empty persona. A single `persona` / `persona_text` counts as one.
    TA agents play each of their personas (all of them, `<id>_p1`, `<id>_p2`, ...), whatever N.
    """
    if agent.personas is not None and (agent.persona or agent.persona_text):
        raise ValueError(f"agent {agent.id}: give `personas` or `persona`/`persona_text`, not both")
    if count is None:
        if agent.personas is None:
            return [agent]
        files = _persona_list(agent)
        if not files:
            raise ValueError(f"agent {agent.id}: personas {agent.personas} match no file")
        return [agent.model_copy(update={"id": f"{agent.id}_{Path(f).stem}", "persona": f,
                                         "personas": None, "variant_of": agent.id,
                                         "personas_given": len(files)})
                for f in files]
    if count < 1:
        raise ValueError("personas_per_student must be at least 1")
    if agent.owner == "ta":
        if agent.personas is None:
            return [agent]
        files = _persona_list(agent)
        if not files:
            raise ValueError(f"agent {agent.id}: personas {agent.personas} match no file")
        return [agent.model_copy(update={"id": f"{agent.id}_p{k + 1}", "persona": f,
                                         "personas": None, "variant_of": agent.id,
                                         "personas_given": len(files)})
                for k, f in enumerate(files)]
    if agent.personas is not None:
        given: list[dict[str, Any]] = [{"persona": f, "persona_text": None}
                                       for f in _persona_list(agent)]
    elif agent.persona or agent.persona_text:
        given = [{"persona": agent.persona, "persona_text": agent.persona_text}]
    else:
        given = []
    chosen = (given[:count] + given[:1] * max(0, count - len(given))) if given else \
        [{"persona": None, "persona_text": None}] * count
    return [agent.model_copy(update={"id": f"{agent.id}_p{k + 1}", **p, "personas": None,
                                     "variant_of": agent.id, "personas_given": len(given)})
            for k, p in enumerate(chosen)]


def expand_agents(patterns: list, root: Path, persona_count: Optional[int] = None) -> list[AgentSpec]:
    """Agent files / globs, or mappings {agent: <file>, personas: [...], lora: <path|null>}
    that set those fields for that file; each agent is expanded per persona."""
    seen: dict[str, AgentSpec] = {}
    for entry in patterns:
        over: dict[str, Any] = {}
        if isinstance(entry, dict):
            unknown = set(entry) - {"agent", "personas", "lora"}
            if unknown or "agent" not in entry:
                raise ValueError(f"agent entry {entry}: needs `agent`, may set `personas` and "
                                 f"`lora` (unknown: {sorted(unknown)})")
            over = {k: entry[k] for k in ("personas", "lora") if k in entry}
            if isinstance(over.get("personas"), str):
                over["personas"] = [over["personas"]]
            entry = entry["agent"]
        matches = sorted(glob.glob(str(root / entry), recursive=True))
        literal = root / entry.replace("[[]", "[").replace("[*]", "*").replace("[?]", "?")
        if not matches and literal.name == "agent.yaml" and is_agent_folder(literal.parent):
            matches = [str(literal)]             # a role folder without agent.yaml
        if not matches:
            raise ValueError(f"no agent files match {entry!r}")
        files = []
        for m in matches:                       # a role folder stands for its agent.yaml
            mp = Path(m)
            if mp.is_dir():
                if (mp / "agent.yaml").exists() or is_agent_folder(mp):
                    files.append(str(mp / "agent.yaml"))
                elif any((mp / r).is_dir() for r in ROLES):
                    raise ValueError(f"{entry!r} is a student folder: name its role folder, "
                                     f"e.g. {entry.rstrip('/')}/seller or {entry.rstrip('/')}/buyer")
                else:
                    raise ValueError(f"{entry!r} is a folder with no agent in it "
                                     f"(no agent.yaml, checkpoint/ or persona files)")
            else:
                files.append(m)
        for m in files:
            agent = load_agent(Path(m))
            if over:
                if "personas" in over:
                    over = {**over, "persona": None, "persona_text": None}
                agent = agent.model_copy(update=over)
            for a in persona_variants(agent, persona_count):
                if a.id in seen:
                    raise ValueError(f"agent id {a.id!r} appears twice ({m})")
                seen[a.id] = a
    return list(seen.values())


class LoadedRun(BaseModel):
    """A run config with every reference resolved and cross-checked."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RunConfig
    root: Path
    models: dict[str, ModelSpec]
    serving: dict[str, ServingSpec] = {}
    sellers: list[AgentSpec]
    buyers: list[AgentSpec]
    # Agents that only top up short markets (SideCasting.pad), per side: "sellers" / "buyers"
    pads: dict[str, list[AgentSpec]] = {}

    def everyone(self) -> list[AgentSpec]:
        """Every agent that may play, padding agents included."""
        return (self.sellers + self.buyers + self.pads.get("sellers", [])
                + self.pads.get("buyers", []))


def load_run(path: str | Path, root: Optional[Path] = None) -> LoadedRun:
    """Load a raw run config (casting spelled out). Relative paths resolve against `root`
    (default: the current directory, i.e. the repo root)."""
    path = Path(path)
    data = _read_yaml(path)
    data["source"] = str(path)
    return resolve_run(RunConfig(**data), root)


def resolve_run(config: RunConfig, root: Optional[Path] = None) -> LoadedRun:
    """Resolve every model and agent a run config refers to, and cross-check them."""
    root = Path(root) if root else Path.cwd()
    models, serving = load_models(config.models, root)
    sellers = expand_agents(config.casting.sellers.agents, root, config.personas_per_student)
    buyers = expand_agents(config.casting.buyers.agents, root, config.personas_per_student)
    pads = {side: expand_agents(getattr(config.casting, side).pad, root, config.personas_per_student)
            for side in ("sellers", "buyers") if getattr(config.casting, side).pad}
    pad_sellers, pad_buyers = pads.get("sellers", []), pads.get("buyers", [])
    for a in sellers + buyers + pad_sellers + pad_buyers:
        if isinstance(a.model, ModelSpec):          # inline model: register it for this agent
            # Named after the served model (readable in logs); agents with the very same
            # settings share the entry, any other clash gets a per-agent name.
            name = a.model.model
            if models.get(name, a.model) != a.model:
                name = f"agent-{a.id}"
            models[name] = a.model
            a.model = name
    for side, agents in (("seller", sellers + pad_sellers), ("buyer", buyers + pad_buyers)):
        for a in agents:
            if a.role != side:
                raise ValueError(f"agent {a.id} ({a.source}) has role {a.role}, cast as {side}")
            if a.controller == "llm" and a.model not in models:
                raise ValueError(f"agent {a.id}: unknown model {a.model!r}; "
                                 f"known: {sorted(models)}")
    for a in sellers + buyers + pad_sellers + pad_buyers:
        if a.lora:
            if a.controller != "llm":
                raise ValueError(f"agent {a.id}: `lora` needs controller: llm")
            base = models[a.model]
            if base.lora:
                raise ValueError(f"agent {a.id}: `model` must name a base model, not a LoRA")
            if base.api_key or base.api_key_env:
                raise ValueError(f"agent {a.id}: a LoRA needs an open-weight base model served "
                                 f"by vLLM, not the hosted API {base.base_url}")
            name = f"lora-{a.variant_of or a.id}"       # persona copies share one adapter
            path = (Path(a.source).parent / a.lora).resolve()
            shared = next((n for n, m in models.items() if m.lora and m.lora.path == str(path)
                           and m.lora.base_model == base.model), None)
            if shared:                          # the same adapter folder: one registration
                a.model = shared
                continue
            if name in models:
                raise ValueError(f"agent {a.id}: LoRA name {name} is already used for another adapter")
            check_adapter(a.id, path, base.model, base.lora_rename,
                          max_rank=TA_MAX_LORA_RANK if a.owner == "ta" else MAX_LORA_RANK)
            models[name] = base.model_copy(update={
                "model": name, "lora": LoraSpec(base_model=base.model, path=str(path))})
            a.model = name
    for a in sellers + buyers + pad_sellers + pad_buyers:
        if a.controller == "llm":
            n = len(a.persona_body())
            if n > config.market.persona_chars:
                raise ValueError(f"agent {a.id}: persona is {n} characters, over the limit of "
                                 f"{config.market.persona_chars} (market.persona_chars)")
    ids = [a.id for a in sellers + buyers]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"agent ids used on both sides: {sorted(dupes)}")
    # Padding agents may also play as regular agents (e.g. TA sellers that are grade anchors
    # and fill short markets): within one market a second seat becomes a copy (`<id>_2`).
    # Two different agent files under one id would be ambiguous, though.
    src = {a.id: a.source for a in sellers + buyers}
    clash = sorted(a.id for a in pad_sellers + pad_buyers if a.id in src and src[a.id] != a.source)
    if clash:
        raise ValueError(f"padding agents share an id with different regular agents: {clash}")
    return LoadedRun(config=config, root=root, models=models, serving=serving,
                     sellers=sellers, buyers=buyers, pads=pads)
