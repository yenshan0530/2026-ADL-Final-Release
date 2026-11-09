"""Market state and rules: one market, its goods, and every action's effect.

All money is integer cents internally; agents see and send dollars. The engine is the only
source of truth for stock, cash, grades, offers and trades: nothing an agent writes in a
message changes them.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from adlmarket.config import AgentSpec, MarketSpec
from adlmarket.schema import Schema
from adlmarket.tools import ROLE_TOOLS, TOOLS, build_tools, dollars, to_cents

GRADES = ("high", "low")


# --------------------------------------------------------------------------- catalog ---

@dataclass(frozen=True)
class Product:
    id: str
    title: str
    category: str
    ref_price: int          # cents; the public "list price" anchor
    brand: str = ""
    product_type: str = ""
    average_rating: Optional[float] = None
    description: str = ""
    summary: str = ""          # one line, written by hand


def load_catalog(path: Path) -> list[Product]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Product(id=p["id"], title=p["title"], category=p.get("category", ""),
                    ref_price=to_cents(p["reference_price"]), brand=p.get("brand") or "",
                    product_type=p.get("product_type") or "",
                    average_rating=p.get("average_rating"),
                    description=p.get("description") or "",
                    summary=p.get("summary") or "")
            for p in data["products"]]


# ---------------------------------------------------------------------------- state ---

@dataclass
class SellerBook:
    id: str
    stock: dict[str, int]
    cost: dict[str, int]                 # high-grade unit cost, private
    low_cost: dict[str, int] = field(default_factory=dict)   # quality-varying products only
    floor: dict[str, int] = field(default_factory=dict)      # engine-enforced minimum price
    revenue: int = 0
    cost_of_sold: int = 0
    initial_stock: dict[str, int] = field(default_factory=dict)


@dataclass
class BuyerBook:
    id: str
    cash: int
    budget: int
    utility: dict[str, tuple[int, ...]]  # utility of the 1st, 2nd, ... unit, private
    held: dict[str, list[str]] = field(default_factory=dict)  # grades of units bought
    spent: int = 0
    gained_utility: int = 0
    max_price_ratio: Optional[float] = None   # engine-enforced: price <= ratio x unit utility

    def next_utility(self, pid: str) -> int:
        k = len(self.held.get(pid, []))
        table = self.utility.get(pid, ())
        return table[k] if k < len(table) else 0


@dataclass
class Listing:
    price: int
    grade: str = "high"                  # what will be delivered; hidden from buyers
    certified: bool = False


@dataclass
class Offer:
    id: str
    seller: str
    buyer: str
    product: str
    price: int
    qty: int
    made_round: int
    expires_after_round: int
    grade: str = "high"
    certified: bool = False
    status: str = "open"                 # open | accepted | expired


@dataclass
class Message:
    id: str
    round: int
    tick: int
    channel: str                         # dm | seller_room | buyer_room | notice | review
    sender: str
    to: Optional[str]                    # dm/notice recipient; None for rooms and reviews
    text: str
    offer_id: Optional[str] = None


@dataclass
class Trade:
    id: str
    round: int
    tick: int
    seller: str
    buyer: str
    product: str
    price: int
    qty: int
    via: str                             # board | offer
    grade: str = "high"
    certified: bool = False
    utility: int = 0                     # utility the buyer got from these units
    reviewed: bool = False


@dataclass
class Swap:
    """A binding trade between two buyers (MarketSpec.buyer_swaps): `proposer` hands over
    `give` and `cash` (cents; negative = receives cash) for `get` from `to`. `give_grades` /
    `get_grades` say, per quality-varying product, how many high- and low-grade units move
    (resolved when the swap is proposed, so both sides know exactly what they get)."""
    id: str
    proposer: str
    to: str
    give: dict[str, int]
    get: dict[str, int]
    cash: int
    made_round: int
    expires_after_round: int
    status: str = "open"                 # open | accepted | expired
    give_grades: dict[str, dict[str, int]] = field(default_factory=dict)
    get_grades: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class Review:
    id: str
    round: int
    trade_id: str
    buyer: str
    seller: str
    product: str
    rating: int
    text: str


@dataclass
class Result:
    ok: bool
    error: Optional[str] = None
    detail: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok}
        if self.error:
            out["error"] = self.error
            out["detail"] = self.detail
        if self.data:
            out["data"] = self.data
        return out


def _fail(error: str, detail: str) -> Result:
    return Result(ok=False, error=error, detail=detail)


def _qty(value: Any, minimum: int = 1) -> int:
    """A quantity from the model: 2, 2.0 or "2" are fine; "two", 2.5 or 0 are not."""
    try:
        q = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"qty must be a whole number, got {value!r}") from None
    if not math.isfinite(q) or q != int(q) or q < minimum:
        raise ValueError(f"qty must be a whole number of at least {minimum}, got {value!r}")
    return int(q)


def _low_last(grade: str) -> bool:
    """Sort key that puts high-grade units ahead of low-grade ones."""
    return grade == "low"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"expected true or false, got {value!r}")


# --------------------------------------------------------------------------- market ---

class Market:
    def __init__(self, market_id: str, spec: MarketSpec, schema: Schema,
                 products: list[Product], sellers: list[SellerBook], buyers: list[BuyerBook]):
        self.id = market_id
        self.spec = spec
        self.schema = schema
        self.products = {p.id: p for p in products}
        self.sellers = {s.id: s for s in sellers}
        self.buyers = {b.id: b for b in buyers}
        self.round = 0
        self.tick = 0
        self.listings: dict[tuple[str, str], Listing] = {}
        # Board prices set this turn: posted by end_turn(), so every buy in a turn is at the
        # price its buyer saw (the order actions are applied in cannot change it).
        self._pending_listings: dict[tuple[str, str], Listing] = {}
        self.offers: dict[str, Offer] = {}
        self.messages: list[Message] = []
        self.trades: list[Trade] = []
        self.reviews: list[Review] = []
        self.swaps: dict[str, Swap] = {}
        self._seen: dict[str, int] = {a: 0 for a in [*self.sellers, *self.buyers]}
        self.has_quality = any(p.quality_varies for p in schema.plans)
        self.quality_products = {p.product_id for p in schema.plans if p.quality_varies}
        # Lists of sellers/buyers are shown to each agent in its own random order, so no
        # one gains from where their id sorts (buyers tend to take the first of equal offers).
        self.seed_id = market_id                 # random streams; a rotation shares its market's
        self._order_rng = random.Random(f"order:{market_id}")
        self.tools = {role: {t["function"]["name"]: t for t in build_tools(role, spec, self.has_quality)}
                      for role in ("seller", "buyer")}

    # ------------------------------------------------------------------ generation ---

    @classmethod
    def generate(cls, market_id: str, spec: MarketSpec, schema: Schema,
                 catalog: dict[str, Product], sellers: list[AgentSpec], buyers: list[AgentSpec],
                 rng: random.Random, market_index: int = 0, rotation: int = 0,
                 seed_id: Optional[str] = None) -> "Market":
        """Stock this market from the schema. Every buyer gets the schema's budget and its
        utility table (or a private scaling of it, see MarketSpec.buyer_values); stock per
        product is supply_ratio x what the buyers want at list price,
        split evenly over the sellers (or over the few that hold a monopoly or oligopoly product).

        Item-spec schemas (schema.bundles set): seat i holds bundle (i + rotation) % sellers,
        i.e. the products whose `slots` contain it; each buyer wants one unit of each product,
        so supply = supply_ratio x buyers; the budget is set here from what is for sale.
        `rotation` also shifts the classic rotation of concentrated products; `seed_id` (the
        market id without its rotation) keys the random streams, so rotations differ only in
        who holds what."""
        seed_id = seed_id or market_id
        n_buyers = len(buyers)
        n = len(sellers)
        if schema.bundles is not None and n != schema.bundles:
            raise ValueError(f"market {market_id}: {n} sellers, but the schema's items are dealt "
                             f"to {schema.bundles} seller bundles (sellers per market must be exact)")
        stock = {s.id: {} for s in sellers}
        seat = 0                                 # next seat to hold a concentrated product
        for_sale = 0                             # item-spec budget: units x mean(cost, value)
        for plan in schema.plans:
            ref = catalog[plan.product_id].ref_price
            if schema.bundles is not None:
                supply = max(1, round(plan.supply_ratio * n_buyers))
                for_sale += supply * (plan.cost + plan.utility[0]) / 2
            else:
                supply = max(1, round(plan.supply_ratio * n_buyers * plan.wanted_at_list(ref)))
            for s in sellers:
                stock[s.id][plan.product_id] = 0
            holders = sellers
            if schema.bundles is not None and plan.slots is not None:
                holders = [sellers[i] for i in range(n) if (i + rotation) % n in plan.slots]
            elif schema.bundles is None and plan.holders and plan.holders < n:
                holders = [sellers[(seat + i + market_index + rotation) % n]   # over seats and markets
                           for i in range(plan.holders)]
                seat += plan.holders
            base, extra = divmod(supply, len(holders))
            lucky = {s.id for s in rng.sample(holders, extra)}
            for s in holders:
                stock[s.id][plan.product_id] = base + (1 if s.id in lucky else 0)

        seller_books = []
        for a in sellers:
            cost = {p.product_id: p.cost for p in schema.plans}
            low = {p.product_id: p.low_cost for p in schema.plans if p.quality_varies}
            floor = ({pid: round(c * a.guards["min_markup"]) for pid, c in cost.items()}
                     if "min_markup" in a.guards else {})
            seller_books.append(SellerBook(id=a.id, stock=stock[a.id], cost=cost, low_cost=low,
                                           floor=floor, initial_stock=dict(stock[a.id])))
        values = random.Random(f"values:{seed_id}")   # own stream: shared runs are unchanged

        def table(plan: Any) -> tuple[int, ...]:
            if spec.buyer_values == "shared":
                return plan.utility
            f = values.uniform(1 - spec.private_value_spread, 1 + spec.private_value_spread)
            return tuple(round(u * f) for u in plan.utility)

        budget = schema.budget
        if schema.budget_extra is not None:
            budget = round(for_sale * (1 + schema.budget_extra) / n_buyers)
        buyer_books = [BuyerBook(id=a.id, cash=budget, budget=budget,
                                 utility={p.product_id: table(p) for p in schema.plans},
                                 max_price_ratio=a.guards.get("max_price_to_utility"))
                       for a in buyers]
        products = [catalog[p.product_id] for p in schema.plans]
        market = cls(market_id, spec, schema, products, seller_books, buyer_books)
        if seed_id != market_id:
            market.seed_id = seed_id
            market._order_rng = random.Random(f"order:{seed_id}")
        return market

    # -------------------------------------------------------------------- helpers ---

    def role_of(self, agent_id: str) -> Optional[str]:
        if agent_id in self.sellers:
            return "seller"
        if agent_id in self.buyers:
            return "buyer"
        return None

    def tool_schemas(self, role: str) -> list[dict[str, Any]]:
        return list(self.tools[role].values())

    def _next_id(self, prefix: str, collection: Any) -> str:
        return f"{prefix}{len(collection) + 1}"

    def _post(self, channel: str, sender: str, to: Optional[str], text: str,
              offer_id: Optional[str] = None) -> Message:
        msg = Message(id=self._next_id("m", self.messages), round=self.round, tick=self.tick,
                      channel=channel, sender=sender, to=to, text=text, offer_id=offer_id)
        self.messages.append(msg)
        return msg

    def notice(self, to: str, text: str) -> None:
        self._post("notice", "market", to, text)

    def _check_message(self, text: Any) -> Optional[Result]:
        if not isinstance(text, str) or not text.strip():
            return _fail("bad_args", "message must be non-empty text")
        if len(text) > self.spec.max_message_chars:
            return _fail("message_too_long",
                         f"message is {len(text)} chars; limit {self.spec.max_message_chars}")
        return None

    def _grade_terms(self, seller: SellerBook, pid: str, grade: Any,
                     certified: Any) -> tuple[Optional[Result], str, bool]:
        """Validate the grade a seller will deliver and whether it is certified."""
        grade = "high" if grade is None else str(grade).strip().lower()
        certified = False if certified is None else _bool(certified)
        if grade not in GRADES:
            return _fail("bad_args", "grade must be 'high' or 'low'"), grade, certified
        if grade == "low" and pid not in seller.low_cost:
            return _fail("bad_args", f"{pid} only comes in high grade"), grade, certified
        if certified and grade != "high":
            return _fail("cannot_certify", "only high-grade units can be certified"), grade, certified
        return None, grade, certified

    # --------------------------------------------------------------------- actions ---

    def apply(self, agent_id: str, tool: str, args: dict[str, Any]) -> Result:
        """Apply one tool call. Invalid calls change nothing and say why."""
        role = self.role_of(agent_id)
        if role is None:
            return _fail("unknown_agent", agent_id)
        allowed = self.tools[role]
        if tool not in TOOLS:
            return _fail("unknown_tool", f"there is no tool {tool!r}; "
                                         f"yours are: {', '.join(sorted(allowed))}")
        if tool not in ROLE_TOOLS[role]:
            return _fail("not_permitted", f"{role}s cannot call {tool!r}")
        if tool not in allowed:
            return _fail("channel_disabled", f"{tool!r} is switched off in this market")
        args = {k: v for k, v in args.items() if k != "private_state"}

        # Arguments are checked against the same schema the model was shown.
        params = allowed[tool]["function"]["parameters"]
        unknown = sorted(set(args) - set(params["properties"]))
        missing = [k for k in params["required"] if k not in args]
        if unknown or missing:
            return _fail("bad_args", "; ".join(
                ([f"unknown argument(s): {', '.join(unknown)}"] if unknown else []) +
                ([f"missing argument(s): {', '.join(missing)}"] if missing else [])))

        try:
            match tool:
                case "set_price":
                    return self._do_set_price(agent_id, args["product_id"], args["price"],
                                              args.get("grade"), args.get("certified"))
                case "make_offer":
                    return self._do_make_offer(agent_id, args["buyer_id"], args["product_id"],
                                               args["price"], args["qty"], args.get("message", ""),
                                               args.get("grade"), args.get("certified"))
                case "buy_listed":
                    return self._do_buy_listed(agent_id, args["seller_id"], args["product_id"],
                                               args["qty"])
                case "accept_offer":
                    return self._do_accept_offer(agent_id, args["offer_id"])
                case "post_review":
                    return self._do_post_review(agent_id, args["trade_id"], args["rating"],
                                                args.get("text", ""))
                case "propose_swap":
                    return self._do_propose_swap(agent_id, args["buyer_id"], args["give"],
                                                 args["get"], args.get("cash", 0),
                                                 args.get("give_grade"), args.get("get_grade"),
                                                 args.get("message", ""))
                case "accept_swap":
                    return self._do_accept_swap(agent_id, args["swap_id"])
                case "send_message":
                    return self._do_send_message(agent_id, args["to"], args["message"])
                case "post_to_room":
                    return self._do_post_to_room(agent_id, args["message"])
                case "wait":
                    return self._do_wait(agent_id)
                case _:
                    return _fail("not_permitted", f"unknown tool {tool!r}")
        except (TypeError, ValueError, ArithmeticError) as exc:     # e.g. qty "two", price -3
            return _fail("bad_args", str(exc))

    def end_turn(self) -> None:
        """Post the board prices set this turn (see _pending_listings)."""
        self.listings.update(self._pending_listings)
        self._pending_listings.clear()

    def _do_wait(self, agent_id: str) -> Result:
        return Result(ok=True)

    def _do_send_message(self, agent_id: str, to: str, message: str) -> Result:
        bad = self._check_message(message)
        if bad:
            return bad
        target = self.role_of(to)
        if target is None or to == agent_id:
            return _fail("unknown_agent", f"no other agent {to!r} in this market")
        if not self.spec.channels.dm_same_side and target == self.role_of(agent_id):
            return _fail("not_permitted", "messages to your own side are switched off")
        msg = self._post("dm", agent_id, to, message)
        return Result(ok=True, data={"message_id": msg.id})

    def _do_post_to_room(self, agent_id: str, message: str) -> Result:
        bad = self._check_message(message)
        if bad:
            return bad
        msg = self._post(f"{self.role_of(agent_id)}_room", agent_id, None, message)
        return Result(ok=True, data={"message_id": msg.id})

    def _do_set_price(self, agent_id: str, product_id: str, price: Any, grade: Any = None,
                      certified: Any = None) -> Result:
        seller = self.sellers[agent_id]
        if product_id not in self.products:
            return _fail("unknown_product", f"{product_id!r}; products: {', '.join(self.products)}")
        cents = to_cents(price)
        if cents < seller.floor.get(product_id, 0):
            return _fail("below_floor", "price is below your enforced minimum")
        bad, grade, certified = self._grade_terms(seller, product_id, grade, certified)
        if bad:
            return bad
        self._pending_listings[(agent_id, product_id)] = Listing(cents, grade, certified)
        data = {"product_id": product_id, "price": dollars(cents),
                "note": "posted on the board at the end of this turn"}
        if product_id in seller.low_cost:
            data.update(grade=grade, certified=certified)
        return Result(ok=True, data=data)

    def _do_make_offer(self, agent_id: str, buyer_id: str, product_id: str, price: Any,
                       qty: Any, message: str = "", grade: Any = None,
                       certified: Any = None) -> Result:
        seller = self.sellers[agent_id]
        if buyer_id not in self.buyers:
            return _fail("unknown_agent", f"no buyer {buyer_id!r} in this market")
        if product_id not in self.products:
            return _fail("unknown_product", f"{product_id!r}; products: {', '.join(self.products)}")
        qty = _qty(qty)
        if seller.stock.get(product_id, 0) < qty:
            return _fail("no_stock", f"you have {seller.stock.get(product_id, 0)} unit(s)")
        cents = to_cents(price)
        if cents < seller.floor.get(product_id, 0):
            return _fail("below_floor", "price is below your enforced minimum")
        bad, grade, certified = self._grade_terms(seller, product_id, grade, certified)
        if bad:
            return bad
        if message:
            bad = self._check_message(message)
            if bad:
                return bad
        open_count = sum(1 for o in self.offers.values()
                         if o.seller == agent_id and o.status == "open")
        if open_count >= self.spec.max_open_offers:
            return _fail("too_many_offers", f"limit {self.spec.max_open_offers} open offers")
        offer = Offer(id=self._next_id("o", self.offers), seller=agent_id, buyer=buyer_id,
                      product=product_id, price=cents, qty=qty, made_round=self.round,
                      expires_after_round=self.round + self.spec.offer_ttl_rounds - 1,
                      grade=grade, certified=certified)
        self.offers[offer.id] = offer
        terms = (f"[binding offer {offer.id}] {qty} x {product_id} at ${dollars(cents):.2f} each"
                 f"{', certified high grade' if certified else ''}, "
                 f"open until the end of round {offer.expires_after_round}")
        self._post("dm", agent_id, buyer_id, f"{terms}. {message}".strip(), offer_id=offer.id)
        return Result(ok=True, data={"offer_id": offer.id})

    def _do_buy_listed(self, agent_id: str, seller_id: str, product_id: str, qty: Any) -> Result:
        if seller_id not in self.sellers:
            return _fail("unknown_agent", f"no seller {seller_id!r} in this market")
        listing = self.listings.get((seller_id, product_id))
        if listing is None:
            return _fail("not_listed", f"{seller_id} has no board price for {product_id}")
        return self._trade(seller_id, agent_id, product_id, listing.price, _qty(qty), via="board",
                           grade=listing.grade, certified=listing.certified)

    def _do_accept_offer(self, agent_id: str, offer_id: str) -> Result:
        offer = self.offers.get(offer_id)
        if offer is None or offer.buyer != agent_id:
            return _fail("offer_not_found", f"no offer {offer_id!r} addressed to you")
        if offer.status != "open":
            return _fail("offer_closed", f"offer {offer_id} is {offer.status}")
        res = self._trade(offer.seller, agent_id, offer.product, offer.price, offer.qty,
                          via="offer", grade=offer.grade, certified=offer.certified)
        if res.ok:
            offer.status = "accepted"
        return res

    def _units(self, units: Any) -> dict[str, int]:
        """{product_id: qty} from the model; a qty of 0 means "no units" and is dropped."""
        if not isinstance(units, dict):
            raise ValueError("give/get must be an object {product_id: qty}")
        out = {}
        for pid, q in units.items():
            if pid not in self.products:
                raise ValueError(f"unknown product {pid!r}")
            if q := _qty(q, minimum=0):
                out[pid] = q
        return out

    def _swap_cash(self, cash: Any) -> int:
        """Cash from the model -> cents: positive = the proposer pays, negative = it receives."""
        if cash is None:
            return 0
        text = str(cash).strip().replace("$", "").replace(",", "")
        neg = text.startswith("-")
        digits = text.lstrip("+-")
        try:
            if float(digits) == 0:               # "0", "0.0", "0.000" all mean no cash
                return 0
            return -to_cents(digits) if neg else to_cents(digits)
        except ValueError as exc:
            raise ValueError(f"cash: {exc}") from None

    def _swap_grades(self, units: dict[str, int], grades: Any, label: str) -> dict[str, str]:
        """{product_id: high|low} from the model, for products this side hands over."""
        if not grades:
            return {}
        if not isinstance(grades, dict):
            raise ValueError(f"{label} must be an object {{product_id: grade}}")
        out = {}
        for pid, grade in grades.items():
            if pid not in units:
                raise ValueError(f"{label}: {pid!r} is not in the units you give/get")
            grade = str(grade).strip().lower()
            if grade not in GRADES:
                raise ValueError(f"{label}: grade must be 'high' or 'low', got {grade!r}")
            if pid not in self.quality_products:
                raise ValueError(f"{label}: {pid} only comes in high grade")
            out[pid] = grade
        return out

    def _swap_breakdown(self, giver: BuyerBook, units: dict[str, int],
                        grades: dict[str, str]) -> dict[str, dict[str, int]]:
        """How many high- and low-grade units of each quality-varying product `giver` hands
        over: the grade it pinned, else its low-grade units first. Whether it holds that many
        is checked by _swap_problem."""
        out = {}
        for pid, q in units.items():
            if pid not in self.quality_products:
                continue
            held = giver.held.get(pid, [])
            if pinned := grades.get(pid):
                out[pid] = {pinned: q}
            else:
                low = min(q, held.count("low"))
                out[pid] = {g: n for g, n in (("low", low), ("high", q - low)) if n}
        return out

    def _swap_problem(self, giver: BuyerBook, units: dict[str, int],
                      grades: dict[str, dict[str, int]], cash: int) -> Optional[str]:
        for pid, q in units.items():
            held = giver.held.get(pid, [])
            if pid in grades:
                for grade, n in grades[pid].items():
                    if held.count(grade) < n:
                        return (f"{giver.id} holds {held.count(grade)} {grade}-grade unit(s) "
                                f"of {pid}, not {n}")
            elif len(held) < q:
                return f"{giver.id} holds {len(held)} unit(s) of {pid}, not {q}"
        if cash > giver.cash:
            return f"{giver.id} has ${dollars(giver.cash):.2f}, not ${dollars(cash):.2f}"
        return None

    def _swap_text(self, w: Swap) -> str:
        def fmt(units: dict[str, int], grades: dict[str, dict[str, int]]) -> str:
            parts = []
            for pid, q in units.items():
                if by_grade := grades.get(pid):
                    mix = "/".join(f"{n} {g}" for g, n in by_grade.items())
                    parts.append(f"{q} x {pid} ({mix})")
                else:
                    parts.append(f"{q} x {pid}")
            return ", ".join(parts) or "no units"
        pay = (f" + ${dollars(w.cash):.2f} from {w.proposer}" if w.cash > 0 else
               f" + ${dollars(-w.cash):.2f} from {w.to}" if w.cash < 0 else "")
        return (f"[swap {w.id}] {w.proposer} gives {fmt(w.give, w.give_grades)}; "
                f"{w.to} gives {fmt(w.get, w.get_grades)}"
                f"{pay}. Open until the end of round {w.expires_after_round}.")

    def _do_propose_swap(self, agent_id: str, to: str, give: Any, get: Any, cash: Any = 0,
                         give_grade: Any = None, get_grade: Any = None,
                         message: str = "") -> Result:
        if to not in self.buyers or to == agent_id:
            return _fail("unknown_agent", f"no other buyer {to!r} in this market")
        give, get = self._units(give), self._units(get)
        give_pin = self._swap_grades(give, give_grade, "give_grade")
        get_pin = self._swap_grades(get, get_grade, "get_grade")
        cents = self._swap_cash(cash)
        if not give and not get and not cents:
            return _fail("bad_args", "a swap must move at least one unit or some cash")
        if message:
            bad = self._check_message(message)
            if bad:
                return bad
        proposer, taker = self.buyers[agent_id], self.buyers[to]
        give_grades = self._swap_breakdown(proposer, give, give_pin)
        get_grades = self._swap_breakdown(taker, get, get_pin)
        problem = (self._swap_problem(proposer, give, give_grades, max(0, cents))
                   or self._swap_problem(taker, get, get_grades, max(0, -cents)))
        if problem:
            return _fail("cannot_give", problem)
        mine = [w for w in self.swaps.values() if w.proposer == agent_id and w.status == "open"]
        if len(mine) >= self.spec.max_open_offers:
            return _fail("too_many_open", f"you already have {len(mine)} open swaps")
        w = Swap(id=self._next_id("w", self.swaps), proposer=agent_id, to=to, give=give, get=get,
                 cash=cents, made_round=self.round,
                 expires_after_round=self.round + self.spec.offer_ttl_rounds - 1,
                 give_grades=give_grades, get_grades=get_grades)
        self.swaps[w.id] = w
        self._post("dm", agent_id, to, self._swap_text(w) + (f" {message}" if message else ""))
        return Result(ok=True, data={"swap_id": w.id})

    def _do_accept_swap(self, agent_id: str, swap_id: str) -> Result:
        w = self.swaps.get(swap_id)
        if w is None or w.to != agent_id:
            return _fail("swap_not_found", f"no swap {swap_id!r} proposed to you")
        if w.status != "open":
            return _fail("swap_closed", f"swap {swap_id} is {w.status}")
        a, b = self.buyers[w.proposer], self.buyers[w.to]
        problem = (self._swap_problem(a, w.give, w.give_grades, max(0, w.cash))
                   or self._swap_problem(b, w.get, w.get_grades, max(0, -w.cash)))
        if problem:
            return _fail("cannot_complete", problem)
        for giver, taker, units, grades in ((a, b, w.give, w.give_grades),
                                            (b, a, w.get, w.get_grades)):
            for pid, q in units.items():
                moved = []
                for grade, n in (grades.get(pid) or {}).items():
                    for _ in range(n):
                        giver.held[pid].remove(grade)
                        moved.append(grade)
                moved += [giver.held[pid].pop() for _ in range(q - len(moved))]
                taker.held.setdefault(pid, []).extend(moved)
                taker.held[pid].sort(key=_low_last)
        a.cash -= w.cash
        b.cash += w.cash
        for x in (a, b):
            x.gained_utility = self._held_utility(x)
        w.status = "accepted"
        line = self._swap_text(w).replace("Open until", "Done; was open until")
        self.notice(a.id, line)
        self.notice(b.id, line)
        return Result(ok=True, data={"swap_id": w.id})

    def _product_utility(self, b: BuyerBook, pid: str) -> int:
        """Utility of the units of one product a buyer holds, independent of the order they
        arrived in: high-grade units fill the most valuable slots of its table, low-grade
        units the next ones (x the low-grade factor); units beyond the table are worth 0."""
        table = b.utility.get(pid, ())
        grades = sorted(b.held.get(pid, []), key=_low_last)   # high first
        return sum(round(table[k] * (self.schema.low_utility if g == "low" else 1.0))
                   for k, g in enumerate(grades) if k < len(table))

    def _next_high_unit(self, b: BuyerBook, pid: str) -> int:
        """What one more high-grade unit of pid would add to the buyer's utility."""
        before = self._product_utility(b, pid)
        b.held.setdefault(pid, []).append("high")
        after = self._product_utility(b, pid)
        b.held[pid].pop()
        return after - before

    def _held_utility(self, b: BuyerBook) -> int:
        """Utility of everything a buyer holds (see _product_utility)."""
        return sum(self._product_utility(b, pid) for pid in b.held)

    def _trade(self, seller_id: str, buyer_id: str, pid: str, price: int, qty: int, via: str,
               grade: str, certified: bool) -> Result:
        seller, buyer = self.sellers.get(seller_id), self.buyers[buyer_id]
        if seller is None:
            return _fail("unknown_agent", f"no seller {seller_id!r}")
        if seller.stock.get(pid, 0) < qty:
            return _fail("no_stock", f"{seller_id} has {seller.stock.get(pid, 0)} unit(s) of {pid}")
        if price * qty > buyer.cash:
            return _fail("insufficient_cash", f"costs ${dollars(price * qty):.2f}, "
                                              f"you have ${dollars(buyer.cash):.2f}")
        held = buyer.held.setdefault(pid, [])
        useful_left = max(0, len(buyer.utility.get(pid, ())) - len(held))
        if not self.spec.scoring.allow_excess_purchase and qty > useful_left:
            return _fail("exceeds_need", f"only {useful_left} more unit(s) of {pid} are worth "
                                         f"anything to you")
        if buyer.max_price_ratio is not None:
            table = buyer.utility.get(pid, ())
            for k in range(len(held), len(held) + qty):
                cap = round(buyer.max_price_ratio * (table[k] if k < len(table) else 0))
                if price > cap:
                    return _fail("above_price_limit",
                                 f"${dollars(price):.2f} is above your limit of "
                                 f"${dollars(cap):.2f} for unit {k + 1} of {pid}")
        before = self._product_utility(buyer, pid)
        held.extend([grade] * qty)
        held.sort(key=_low_last)
        gained = self._product_utility(buyer, pid) - before     # marginal gain of these units
        unit_cost = seller.low_cost[pid] if grade == "low" else seller.cost[pid]
        seller.stock[pid] -= qty
        seller.revenue += price * qty
        seller.cost_of_sold += unit_cost * qty
        buyer.cash -= price * qty
        buyer.spent += price * qty
        buyer.gained_utility += gained
        trade = Trade(id=self._next_id("t", self.trades), round=self.round, tick=self.tick,
                      seller=seller_id, buyer=buyer_id, product=pid, price=price, qty=qty,
                      via=via, grade=grade, certified=certified, utility=gained)
        self.trades.append(trade)
        line = f"Trade {trade.id}: {buyer_id} bought {qty} x {pid} from {seller_id} at ${dollars(price):.2f}"
        varies = pid in seller.low_cost
        self.notice(seller_id, line + (f" (you delivered {grade} grade)" if varies else ""))
        self.notice(buyer_id, line + (f". Grade received: {grade}" if varies else ""))
        if varies and self.spec.schema_.quality.disclosure == "public":
            for other in [*self.sellers, *self.buyers]:
                if other not in (seller_id, buyer_id):
                    self.notice(other, f"{line}. Grade delivered: {grade}")
        return Result(ok=True, data={"trade_id": trade.id, "total": dollars(price * qty),
                                     **({"grade": grade} if varies else {})})

    def _do_post_review(self, agent_id: str, trade_id: str, rating: Any, text: str = "") -> Result:
        trade = next((t for t in self.trades if t.id == trade_id), None)
        if trade is None or trade.buyer != agent_id:
            return _fail("trade_not_found", f"no trade {trade_id!r} of yours")
        if trade.reviewed:
            return _fail("already_reviewed", f"you already reviewed {trade_id}")
        r = _qty(rating)
        if r > 5:
            raise ValueError("rating must be 1 to 5")
        if text:
            bad = self._check_message(text)
            if bad:
                return bad
        review = Review(id=self._next_id("r", self.reviews), round=self.round, trade_id=trade_id,
                        buyer=agent_id, seller=trade.seller, product=trade.product,
                        rating=r, text=text)
        self.reviews.append(review)
        trade.reviewed = True
        self._post("review", agent_id, None,
                   f"[review of {trade.seller}, {trade.product}, trade {trade_id}: {r}/5] {text}".strip())
        return Result(ok=True, data={"review_id": review.id})

    # ---------------------------------------------------------------------- rounds ---

    def end_round(self) -> None:
        for o in [*self.offers.values(), *self.swaps.values()]:
            if o.status == "open" and o.expires_after_round <= self.round:
                o.status = "expired"

    # ----------------------------------------------------------------- observation ---

    def visible(self, agent_id: str, msg: Message) -> bool:
        if msg.channel in ("dm", "notice"):
            return msg.to == agent_id or msg.sender == agent_id
        if msg.channel == "review":
            return True
        return msg.channel == f"{self.role_of(agent_id)}_room"

    def seller_rating(self, seller_id: str) -> Optional[dict[str, Any]]:
        rs = [r.rating for r in self.reviews if r.seller == seller_id]
        return {"average": round(sum(rs) / len(rs), 2), "reviews": len(rs)} if rs else None

    def board(self) -> list[dict[str, Any]]:
        rows = []
        for (s, p), listing in sorted(self.listings.items()):
            if self.sellers[s].stock.get(p, 0) <= 0:
                continue
            row: dict[str, Any] = {"seller": s, "product_id": p, "price": dollars(listing.price),
                                   "in_stock": self.sellers[s].stock[p]}
            if self.spec.schema_.quality.certification and p in self.sellers[s].low_cost:
                row["certified_high"] = listing.certified
            if self.spec.schema_.quality.reviews and (rating := self.seller_rating(s)):
                row["seller_rating"] = rating
            rows.append(row)
        return rows

    def overview(self, live: bool = False) -> list[dict[str, Any]]:
        """Market-wide totals per product. The same for every agent."""
        rows = []
        for plan in self.schema.plans:
            pid = plan.product_id
            ref = self.products[pid].ref_price
            row: dict[str, Any] = {
                "product_id": pid,
                "units_for_sale": sum(s.initial_stock.get(pid, 0) for s in self.sellers.values()),
                "sellers_with_stock": sum(1 for s in self.sellers.values() if s.initial_stock.get(pid)),
                "units_buyers_want_at_list_price": len(self.buyers) * plan.wanted_at_list(ref),
            }
            if self.has_quality:
                row["quality_varies"] = plan.quality_varies
            if live:
                row["units_still_for_sale"] = sum(s.stock.get(pid, 0) for s in self.sellers.values())
            rows.append(row)
        return rows

    def observe(self, agent_id: str, *, turns_left: int, mark_seen: bool = True) -> dict[str, Any]:
        """What one agent sees at the start of its turn. Private numbers are its own only."""
        role = self.role_of(agent_id)
        new = [m for m in self.messages[self._seen[agent_id]:]
               if self.visible(agent_id, m) and m.sender != agent_id]
        if mark_seen:
            self._seen[agent_id] = len(self.messages)
        obs: dict[str, Any] = {
            "round": self.round,
            "turns_left_this_round": turns_left,
            "new_messages": [{"id": m.id, "channel": m.channel, "from": m.sender,
                              "text": m.text} for m in new],
        }
        if self.spec.channels.board:
            obs["board"] = self.board()
            self._order_rng.shuffle(obs["board"])
        if self.spec.market_overview == "live":
            obs["market_overview"] = self.overview(live=True)
        if role == "seller":
            s = self.sellers[agent_id]
            obs["you"] = {"stock": s.stock, "profit_so_far": dollars(s.revenue - s.cost_of_sold)}
            obs["your_open_offers"] = [self._offer_view(o, seller_view=True) for o in self.offers.values()
                                       if o.seller == agent_id and o.status == "open"]
        else:
            b = self.buyers[agent_id]
            obs["you"] = {
                "cash": dollars(b.cash),
                "units_bought": {p: len(g) for p, g in b.held.items() if g},
                "utility_of_next_unit": {p: dollars(v) for p in b.utility
                                         if (v := self._next_high_unit(b, p)) > 0},
                "utility_so_far": dollars(b.gained_utility),
            }
            if self.has_quality:
                obs["you"]["units_by_grade"] = {
                    p: {g: gs.count(g) for g in GRADES if gs.count(g)}
                    for p, gs in b.held.items() if gs and p in self.quality_products}
            obs["offers_to_you"] = [self._offer_view(o) for o in self.offers.values()
                                    if o.buyer == agent_id and o.status == "open"]
            if self.spec.buyer_swaps:
                def view(w: Swap) -> dict[str, Any]:
                    # Read from this buyer's side: cash_to_you > 0 means it receives that much.
                    mine = w.to == agent_id
                    out: dict[str, Any] = {
                        "swap_id": w.id, "from": w.proposer, "to": w.to,
                        "they_give" if mine else "you_give": w.give,
                        "you_give" if mine else "they_give": w.get,
                        "cash_to_you": dollars(w.cash if mine else -w.cash),
                        "open_until_round": w.expires_after_round}
                    for key, grades in (("they_give_grades" if mine else "you_give_grades",
                                         w.give_grades),
                                        ("you_give_grades" if mine else "they_give_grades",
                                         w.get_grades)):
                        if grades:
                            out[key] = grades
                    return out
                obs["swaps_to_you"] = [view(w) for w in self.swaps.values()
                                       if w.to == agent_id and w.status == "open"]
                obs["your_open_swaps"] = [view(w) for w in self.swaps.values()
                                          if w.proposer == agent_id and w.status == "open"]
            if self.spec.schema_.quality.reviews and self.has_quality:
                obs["your_unreviewed_trades"] = [
                    {"trade_id": t.id, "seller": t.seller, "product_id": t.product,
                     **({"grade": t.grade} if t.product in self.sellers[t.seller].low_cost else {})}
                    for t in self.trades if t.buyer == agent_id and not t.reviewed]
        return obs

    def _offer_view(self, o: Offer, seller_view: bool = False) -> dict[str, Any]:
        view = {"offer_id": o.id, "seller": o.seller, "buyer": o.buyer, "product_id": o.product,
                "price": dollars(o.price), "qty": o.qty, "open_until_round": o.expires_after_round}
        if o.product in self.sellers[o.seller].low_cost:
            if self.spec.schema_.quality.certification:
                view["certified_high"] = o.certified
            if seller_view:
                view["grade"] = o.grade
        return view

    def _product_view(self, p: Product) -> dict[str, Any]:
        view: dict[str, Any] = {"product_id": p.id, "title": p.title}
        if p.brand:
            view["brand"] = p.brand
        view["type"] = p.product_type or p.category
        view["list_price"] = dollars(p.ref_price)
        if p.average_rating is not None:
            view["rating"] = f"{p.average_rating:.1f}/5"
        if self.spec.product_text == "summary" and p.summary:
            view["summary"] = p.summary
            return view
        n = self.spec.description_chars
        if self.spec.product_text != "none" and p.description and n != 0:
            view["description"] = p.description if n is None or len(p.description) <= n \
                else p.description[:n].rstrip() + "..."
        return view

    def _shuffled(self, items: list[str], key: str) -> list[str]:
        random.Random(f"{self.seed_id}:{key}").shuffle(items)
        return items

    def situation(self, agent_id: str) -> dict[str, Any]:
        """Fixed facts for the system prompt: public market facts plus the agent's own."""
        role = self.role_of(agent_id)
        sit: dict[str, Any] = {
            "your_id": agent_id,
            "products": [self._product_view(p) for p in self.products.values()],
            "sellers": self._shuffled(sorted(self.sellers), f"sellers:{agent_id}"),
            "buyers": self._shuffled(sorted(self.buyers), f"buyers:{agent_id}"),
            "rounds": self.spec.rounds.total,
            "turns_per_round": self.spec.rounds.turns_per_round,
        }
        if self.spec.market_overview != "none":
            sit["market_overview"] = self.overview()
        if role == "seller":
            s = self.sellers[agent_id]
            sit["your_stock"] = s.stock
            sit["your_unit_cost"] = {p: dollars(c) for p, c in s.cost.items()}
            if s.low_cost:
                sit["your_low_grade_unit_cost"] = {p: dollars(c) for p, c in s.low_cost.items()}
            if s.floor:
                sit["your_minimum_price"] = {p: dollars(c) for p, c in s.floor.items()}
        else:
            b = self.buyers[agent_id]
            sit["your_utility_per_unit"] = {p: [dollars(u) for u in table]
                                            for p, table in b.utility.items()}
            if self.has_quality:
                sit["low_grade_utility"] = (f"a low-grade unit is worth {self.schema.low_utility:g} "
                                            f"x the utility above")
            sit["your_budget"] = dollars(b.budget)
            if b.max_price_ratio is not None:
                sit["your_price_limit"] = (f"the market blocks any purchase priced above "
                                           f"{b.max_price_ratio:g} x your utility for that unit")
        return sit

    # --------------------------------------------------------------------- export ---

    def snapshot(self) -> dict[str, Any]:
        return {
            "market_id": self.id,
            "schema": self.schema.describe(),
            "products": [asdict(p) for p in self.products.values()],
            "sellers": [asdict(s) for s in self.sellers.values()],
            "buyers": [asdict(b) for b in self.buyers.values()],
            "listings": [{"seller": s, "product": p, **asdict(l)}
                         for (s, p), l in sorted(self.listings.items())],
            "offers": [asdict(o) for o in self.offers.values()],
            "trades": [asdict(t) for t in self.trades],
            "swaps": [asdict(w) for w in self.swaps.values()],
            "reviews": [asdict(r) for r in self.reviews],
            "messages": [asdict(m) for m in self.messages],
        }
