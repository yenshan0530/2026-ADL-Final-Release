"""Schemas: the items, prices, kinds and seller bundles one economy is played over.

A schema comes from the economy's item file (the items with a `kind`) and is shared by
every market of the economy: all buyers get the same values and budget rule, all sellers
of an item the same unit cost. Only the number of agents differs between markets, and
stock scales with the number of buyers so each item keeps its supply/demand ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


from adlmarket.config import SchemaSpec
from adlmarket.tools import dollars, to_cents


@dataclass(frozen=True)
class ProductPlan:
    product_id: str
    kind: str                       # a key of SchemaSpec.kinds
    supply_ratio: float
    holders: Optional[int]          # how many sellers stock it (1 = monopoly); None = all
    utility: tuple[int, ...]        # cents; utility of the 1st, 2nd, ... unit to every buyer
    cost: int                       # cents; high-grade unit cost, same for every seller
    quality_varies: bool = False
    low_cost: Optional[int] = None  # cents; low-grade unit cost
    # Item-spec schemas: the seller bundles (seat slots) that stock it; None = every seller
    slots: Optional[tuple[int, ...]] = None

    @property
    def one_seller(self) -> bool:
        return self.holders == 1

    def wanted_at_list(self, list_price: int) -> int:
        """Units one buyer would buy at list price: those worth at least the price."""
        return sum(1 for u in self.utility if u >= list_price)


@dataclass(frozen=True)
class Schema:
    id: str
    plans: tuple[ProductPlan, ...]
    budget: int                     # cents; every buyer's budget (0: set per market, below)
    low_utility: float              # low-grade unit utility, x the high-grade utility
    # Item-spec schemas (SchemaSpec.items): sellers per market the bundles are made for, and
    # the budget rule: cash = (1 + budget_extra) x sum(units x mean(cost, value)) / buyers
    bundles: Optional[int] = None
    budget_extra: Optional[float] = None

    def plan(self, product_id: str) -> ProductPlan:
        return next(p for p in self.plans if p.product_id == product_id)

    def describe(self) -> dict[str, Any]:
        return schema_to_dict(self)


def load_item_spec(path: Path) -> dict[str, dict[str, Any]]:
    """{item id: {seller_cost, buyer_value, ...}} from an item spec (scripts/make_item_spec.py)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {it["id"]: it for it in data["items"]}
    if len(items) != len(data["items"]):
        raise ValueError(f"{path}: an item id appears twice")
    return items


def _deal_slots(n_items: int, holders: int, n_bundles: int, start: int) -> list[tuple[int, ...]]:
    """Give each of n_items items `holders` consecutive bundles, cycling, so every bundle
    gets as many as possible: 5 items x 2 holders over 5 bundles -> 2 per bundle."""
    out = []
    pos = start
    for _ in range(n_items):
        out.append(tuple(sorted({(pos + k) % n_bundles for k in range(holders)})))
        pos += holders
    return out


def build_items_schema(schema_id: str, spec: SchemaSpec, items_file: str, catalog: list,
                       root: Path, n_sellers: int) -> Schema:
    """The economy written in the item file: its items (those with a `kind`), their prices
    and low grades. Items sold by some sellers are dealt to `n_sellers` bundles (one per
    seller seat) in file order, as evenly as possible. Nothing is random."""
    items = load_item_spec(root / items_file)
    in_catalog = {p.id for p in catalog}
    played = [it for it in items.values() if it.get("kind")]
    if not played:
        raise ValueError(f"{items_file}: no item has a `kind`, so the economy has no items")
    unknown = sorted({it["kind"] for it in played} - set(spec.kinds))
    if unknown:
        raise ValueError(f"{items_file}: kind(s) {unknown} are not under market.schema.kinds "
                         f"({', '.join(spec.kinds)})")
    missing = [it["id"] for it in played if it["id"] not in in_catalog]
    if missing:
        raise ValueError(f"{items_file}: item(s) {missing} are not in the catalog")
    plans = []
    start = 0
    for kind_name, kind in spec.kinds.items():      # config order: deterministic dealing
        group = [it for it in played if it["kind"] == kind_name]
        holders = kind.holders_in(n_sellers)
        slots: list = [None] * len(group)
        if holders is not None and holders < n_sellers:
            slots = _deal_slots(len(group), holders, n_sellers, start)
            start = (start + len(group) * holders) % n_sellers
        for it, sl in zip(group, slots):
            cost = to_cents(it["seller_cost"])
            varies = bool(it.get("low_grade"))
            plans.append(ProductPlan(
                product_id=it["id"], kind=kind_name, supply_ratio=kind.supply_ratio,
                holders=holders if holders is None or holders < n_sellers else None,
                utility=(to_cents(it["buyer_value"]),), cost=cost, quality_varies=varies,
                low_cost=max(1, round(cost * spec.quality.low_cost)) if varies else None,
                slots=sl))
    order = {it["id"]: i for i, it in enumerate(played)}
    plans.sort(key=lambda p: order[p.product_id])
    return Schema(id=schema_id, plans=tuple(plans), budget=0, low_utility=spec.quality.low_utility,
                  bundles=n_sellers, budget_extra=spec.budget_extra)


# --------------------------------------------------------------------------- dict ---

def schema_to_dict(schema: Schema, provenance: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": schema.id,
        # item-spec schemas: no fixed budget, it is set per market from what is for sale
        "budget": dollars(schema.budget) if schema.budget_extra is None else None,
        **({"bundles": schema.bundles, "budget_extra": schema.budget_extra}
           if schema.bundles is not None else {}),
        "low_utility": schema.low_utility,
        "products": [{
            "product_id": p.product_id,
            "kind": p.kind,
            "supply_ratio": p.supply_ratio,
            "one_seller": p.one_seller,
            **({"sellers": p.holders} if p.holders and p.holders > 1 else {}),
            "utility": [dollars(u) for u in p.utility],
            "cost": dollars(p.cost),
            "quality_varies": p.quality_varies,
            "low_cost": dollars(p.low_cost) if p.low_cost else None,
            **({"slots": list(p.slots)} if p.slots is not None else {}),
        } for p in schema.plans],
    }
    if provenance:
        out["sampled"] = provenance
    return out
