"""Seat assignment: which agents play together in which market.

The partitioned (graded) side is split into markets afresh each repeat, so an agent's
score averages over different opponents and neighbours. The sampled side is drawn per
market, balanced so each agent sits about equally often. With no partitioned side, each
repeat is a single market.

How many markets (Casting.market_count):
  min (default) -- per_market is the smallest market: floor(n / per_market) markets for
                   each partitioned side, and the fewest of those (at least 1). 23 students
                   at per_market 5 make 4 markets of 6, 6, 6 and 5. With both sides
                   partitioned (stage 3), 30 sellers at 5 and 50 buyers at 10 make
                   min(6, 5) = 5 markets of 6 sellers and 10 buyers.
  max           -- per_market is the largest market: ceil(n / per_market) markets, the most
                   of those. 23 students at per_market 10 make 8, 8 and 7.
Either way the split is as even as possible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from adlmarket.config import AgentSpec, Casting, SideCasting


@dataclass
class MarketSeats:
    market_id: str
    repeat: int
    index: int                         # market number within its repeat
    sellers: list[AgentSpec]
    buyers: list[AgentSpec]
    rotation: int = 0                  # Casting.rotate: seat i holds bundle (i + rotation) % n
    seed_id: str = ""                  # random streams: the same for every rotation of a market

    def __post_init__(self) -> None:
        self.seed_id = self.seed_id or self.market_id


def _split(agents: list[AgentSpec], n: int, rng: random.Random) -> list[list[AgentSpec]]:
    """Deal into n near-equal groups at random. Persona copies of one agent file
    (`variant_of`) go to different markets whenever there are at least as many markets as
    copies, and each agent file's markets are drawn at random, so a student's copies do not
    keep meeting the same other students. Market sizes are fixed first (within one of each
    other); then each agent file, in random order (most copies first), puts its copies in
    distinct markets among those with the most room left (ties broken at random), which
    always fills every market exactly."""
    order = agents[:]
    rng.shuffle(order)
    if all(a.variant_of is None for a in agents):
        return [order[i::n] for i in range(n)]
    by_base: dict[str, list[AgentSpec]] = {}
    for a in order:
        by_base.setdefault(a.variant_of or a.id, []).append(a)
    q, extra = divmod(len(agents), n)
    room = [q + 1] * extra + [q] * (n - extra)
    rng.shuffle(room)
    groups: list[list[AgentSpec]] = [[] for _ in range(n)]
    bases = list(by_base)
    rng.shuffle(bases)
    bases.sort(key=lambda b: -len(by_base[b]))          # stable: random order within a size
    for b in bases:
        copies = by_base[b]
        markets = list(range(n))
        rng.shuffle(markets)
        markets.sort(key=lambda m: -room[m])            # most room first, ties random
        for i, a in enumerate(copies):
            m = markets[i % n]                          # distinct while copies <= markets
            groups[m].append(a)
            room[m] -= 1
    return groups


def _sample(agents: list[AgentSpec], n_markets: int, k: int,
            rng: random.Random) -> list[list[AgentSpec]]:
    """k agents per market, cycling through a shuffled deck for balance.

    When k exceeds the pool (e.g. 10 buyer seats, 8 TA personas), personas repeat within
    a market as copies with their own id (`ta_buyer_busy_2`): TA personas are templates,
    and each copy gets its own goods and conversation."""
    deck: list[AgentSpec] = []
    groups = []
    for _ in range(n_markets):
        group: list[AgentSpec] = []
        while len(group) < k:
            if not deck:
                deck = agents[:]
                rng.shuffle(deck)
            used = {a.id for a in group}
            pick = next((a for a in deck if a.id not in used), deck[0])
            deck.remove(pick)
            copies = sum(1 for a in group if a.id == pick.id
                         or (a.id.startswith(pick.id + "_") and a.id[len(pick.id) + 1:].isdigit()))
            group.append(pick if pick.id not in used else
                         pick.model_copy(update={"id": f"{pick.id}_{copies + 1}"}))
        groups.append(group)
    return groups


def _fill(side: SideCasting, agents: list[AgentSpec], n_markets: int,
          rng: random.Random) -> list[list[AgentSpec]]:
    if side.mode == "partition":
        return _split(agents, n_markets, rng)
    if side.mode == "all":
        return [agents[:] for _ in range(n_markets)]
    return _sample(agents, n_markets, side.per_market, rng)


def markets_per_repeat(casting: Casting, sellers: list[AgentSpec], buyers: list[AgentSpec]) -> int:
    """Markets per repeat (economy); see the module docstring."""
    sides = [(side, agents) for side, agents in ((casting.sellers, sellers), (casting.buyers, buyers))
             if side.mode == "partition"]
    if not sides:
        return 1
    exact = [(side, agents) for side, agents in sides if side.exact]
    if exact:                                   # fixed size: enough markets for everyone
        return max(math.ceil(len(agents) / side.per_market) for side, agents in exact)
    if casting.market_count == "max":
        return max(math.ceil(len(agents) / side.per_market) for side, agents in sides)
    return max(1, min(len(agents) // side.per_market for side, agents in sides))


def _pad(side: SideCasting, groups: list[list[AgentSpec]], pool: list[AgentSpec],
         rng: random.Random, name: str) -> None:
    """Top up every group to per_market with agents from `pool`: on an exact-size side every
    group, otherwise only groups smaller than per_market (larger ones stay as they are)."""
    short = [side.per_market - len(g) for g in groups]
    if side.exact and any(n < 0 for n in short):
        raise ValueError(f"{name}: a market has more than per_market={side.per_market} agents")
    short = [max(0, n) for n in short]
    if not any(short):
        return
    if not pool:
        raise ValueError(f"{name}: {sum(short)} seat(s) to fill in short markets but no `pad` "
                         f"agents (e.g. TA {name}) are listed")
    deck: list[AgentSpec] = []
    for g, n in zip(groups, short):
        for _ in range(n):
            if not deck:
                deck = pool[:]
                rng.shuffle(deck)
            used = {a.id for a in g}
            pick = next((a for a in deck if a.id not in used), deck[0])
            deck.remove(pick)
            if pick.id in used:                       # pool smaller than the gap: a copy
                k = 2
                while f"{pick.id}_{k}" in used:
                    k += 1
                pick = pick.model_copy(update={"id": f"{pick.id}_{k}"})
            g.append(pick)


def plan_markets(casting: Casting, sellers: list[AgentSpec], buyers: list[AgentSpec],
                 seed: int, pads: dict[str, list[AgentSpec]] | None = None) -> list[MarketSeats]:
    rng = random.Random(f"casting:{seed}")
    pad_rng = random.Random(f"padding:{seed}")       # own stream: unpadded plans are unchanged
    n_markets = markets_per_repeat(casting, sellers, buyers)
    pads = pads or {}
    plans = []
    for r in range(casting.repeats):
        s_groups = _fill(casting.sellers, sellers, n_markets, rng)
        b_groups = _fill(casting.buyers, buyers, n_markets, rng)
        for side, groups, name in ((casting.sellers, s_groups, "sellers"),
                                   (casting.buyers, b_groups, "buyers")):
            if side.mode == "partition" and (side.exact or pads.get(name)):
                _pad(side, groups, pads.get(name, []), pad_rng, name)
        for i in range(n_markets):
            if not (s_groups[i] and b_groups[i]):
                continue
            base = f"e{r}_m{i}"
            n_rot = len(s_groups[i]) if casting.rotate and len(s_groups[i]) > 1 else 1
            for k in range(n_rot):
                plans.append(MarketSeats(f"{base}_r{k}" if n_rot > 1 else base, r, i,
                                         s_groups[i], b_groups[i], rotation=k, seed_id=base))
    return plans
