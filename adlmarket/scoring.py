"""Per-agent scores, raw and normalized so different schemas are comparable.

seller: profit = revenue - cost of the units (and grades) delivered, plus salvage of unsold
        units if configured. Normalized by selling every unit of its stock at the highest
        utility any buyer has for a first unit of that product (the most it could extract).
buyer:  surplus = utility of the units bought - money spent (= utility + cash left - budget).
        Normalized by the best basket at sellers' high-grade cost within the budget (the
        most it could keep if sellers sold at cost and stock were unlimited).
So 1.0 means "captured all the gains from trade on my side"; 0 means nothing gained;
below 0 means money lost. Within one economy every buyer has the same table and budget,
so raw buyer scores are directly comparable there.

These normalizations are a first proposal, not final grading.
"""

from __future__ import annotations

from typing import Any

from adlmarket.market import Market
from adlmarket.tools import dollars


def best_basket(market: Market, buyer_id: str) -> int:
    """Upper bound on a buyer's surplus: units bought at cost, best surplus-per-dollar first,
    until the budget runs out."""
    b = market.buyers[buyer_id]
    units = []
    for plan in market.schema.plans:
        for u in b.utility[plan.product_id]:
            if u > plan.cost:
                units.append((u - plan.cost, plan.cost))
    units.sort(key=lambda x: x[0] / x[1], reverse=True)
    budget, total = b.budget, 0
    for gain, cost in units:
        if cost <= budget:
            budget -= cost
            total += gain
    return total


def score_market(market: Market) -> list[dict[str, Any]]:
    top_first_unit = {p.product_id: p.utility[0] for p in market.schema.plans}
    salvage = market.spec.scoring.unsold_salvage
    rows = []
    for s in market.sellers.values():
        leftover = sum(round(s.cost[p] * salvage) * q for p, q in s.stock.items())
        profit = s.revenue - s.cost_of_sold + leftover
        benchmark = sum(max(0, top_first_unit[p] - s.cost[p]) * q for p, q in s.initial_stock.items())
        sold = [t for t in market.trades if t.seller == s.id]
        rows.append({"agent": s.id, "role": "seller", "raw": dollars(profit),
                     "normalized": round(profit / benchmark, 4) if benchmark > 0 else None,
                     "units_traded": sum(t.qty for t in sold),
                     "low_grade_units": sum(t.qty for t in sold if t.grade == "low")})
    for b in market.buyers.values():
        surplus = b.gained_utility + b.cash - b.budget    # = utility - spent, plus net swap cash
        benchmark = best_basket(market, b.id)
        bought = [t for t in market.trades if t.buyer == b.id]
        rows.append({"agent": b.id, "role": "buyer", "raw": dollars(surplus),
                     "normalized": round(surplus / benchmark, 4) if benchmark > 0 else None,
                     "units_traded": sum(t.qty for t in bought),
                     "low_grade_units": sum(t.qty for t in bought if t.grade == "low"),
                     "cash_left": dollars(b.cash)})
    return rows
