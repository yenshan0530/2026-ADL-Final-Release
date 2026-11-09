"""System prompt layout, identical for TA and student agents.

    [market rules: TA-owned, same for everyone of a role]
    <<<PERSONA_SECTION>>>
    [persona: the student's text, or a TA persona]
    <<<END_PERSONA_SECTION>>>
    [your situation: this market's private facts, JSON]

Tools travel as native function schemas, not prompt text. Each turn adds one user message
with the observation JSON.
"""

from __future__ import annotations

import json
from typing import Any, Optional

PERSONA_OPEN = "<<<PERSONA_SECTION>>>"
PERSONA_CLOSE = "<<<END_PERSONA_SECTION>>>"

_COMMON = """\
You are an autonomous agent in a simulated marketplace.

ROLE: {role}

HOW A TURN WORKS:
- Each turn you receive an observation (JSON) and respond with exactly ONE function call.
  Text outside a function call is ignored.
- An invalid call does nothing and still uses your turn; the reason is shown next turn.
- `private_state` (optional on every function) is a note to your future self. You see it
  again next turn. Only the last {history} turns are shown to you in full, so keep
  anything important there.
- Prices are US dollars with at most two decimals. Quantities are positive integers.
- The market engine owns stock, cash, offers and trades. Writing something in a message
  does not make it true or binding.
- The market lasts exactly the number of rounds stated in your situation (`rounds`), then
  closes. Anything unsold or unbought when it closes is lost.
"""

_ROLE = {
    "seller": """\
AS A SELLER:
- Your score is your profit: sum of (sale price - your unit cost) over units sold.
  Unsold stock is worth nothing at the end.
{sell_ways}- Your unit costs are private. Every seller of a product has the same unit cost.
""",
    "buyer": """\
AS A BUYER:
{buyer_values}- Your score is the utility of the units you buy plus the cash you have left, minus your
  starting budget: sum of (utility - price paid) over everything you buy. Units with no
  utility left in your table earn nothing.
- Your table gives the utility of your 1st, 2nd, ... unit of each product; each further
  unit is worth less. Your budget does not cover everything worth buying: choose.
{buy_ways}""",
}

_WAYS = {
    True: {"sell_ways": """\
- Two ways to sell: a public board price (`set_price`; it is posted at the end of the turn,
  then anyone may buy at it),
  or a binding private offer to one buyer (`make_offer`). A binding offer cannot be
  withdrawn; it stays open until the round shown on it.
""", "buy_ways": """\
- Two ways to buy: at a seller's public board price (`buy_listed`), or by accepting a
  binding offer a seller made to you (`accept_offer`). A price a seller writes in a
  message is not binding: ask them to send it as an offer.
"""},
    False: {
        "sell_ways": """\
- There is no public price board. You sell only by binding private offers to one buyer
  (`make_offer`); talk to buyers to find out what they would pay. A binding offer cannot
  be withdrawn; it stays open until the round shown on it.
""",
        "buy_ways": """\
- There is no public price board. You buy only by accepting a binding offer a seller made
  to you (`accept_offer`); ask sellers for offers. A price a seller writes in a message is
  not binding: ask them to send it as an offer.
""",
    },
}
_VALUES = {
    "shared": "- Every buyer in this market has the same utility table and the same budget.\n",
    "private": ("- Every buyer has the same budget, but utilities are private and differ between "
                "buyers:\n  each buyer values each product differently. Sellers do not know your "
                "table.\n"),
}

_SWAPS = """\
BUYER-TO-BUYER TRADES:
- You may trade units you hold, and cash, with other buyers: `propose_swap` makes a
  BINDING proposal ({{product_id: qty}} you give and get, plus optional `cash` in dollars:
  positive = you pay them, negative = they pay you; cash alone is allowed); the other buyer
  completes it with `accept_swap` while it is open. A unit is worth to you what your
  utility table says for your k-th unit of that product, whoever you got it from.
{grades}{excess}"""
_GRADES = ("- For products whose quality varies you see `units_by_grade` in your observation, "
           "and a swap shows the grade\n  of every unit that moves. `give_grade` / `get_grade` "
           "({product_id: high|low}) pin the grade; without\n  one you hand over your low-grade "
           "units first.\n")
_EXCESS = ("- You may buy more units than are worth anything to you; units beyond your table "
           "earn\n  nothing, so their price is simply lost.\n")

_OVERVIEW = """\
- The market overview in your situation shows, per product, how many units are for sale,
  how many sellers have stock, and how many units the buyers want at list price. Every
  agent in the market sees the same overview.
"""

_QUALITY = {
    "seller": """\
QUALITY:
- Products marked quality_varies can be delivered in high or low grade. You choose the
  grade (`grade` on `set_price` / `make_offer`; default high). A low-grade unit costs you
  less (your_low_grade_unit_cost). Buyers do not see the grade before buying; the buyer
  learns it when the trade happens.
{certification}{reviews}""",
    "buyer": """\
QUALITY:
- Products marked quality_varies may be delivered in high or low grade. You learn the
  grade only after buying. A low-grade unit gives you only part of its utility (see
  low_grade_utility).
{certification}{reviews}""",
}
_CERT = ("- A listing or offer marked certified_high is guaranteed high grade by the market "
         "engine.\n")
_BUDGET = ("- You may generate at most {n} tokens per round, thinking included, over all your turns;\n"
           "  `output_tokens_left_this_round` in your observation shows what is left. A reply\n"
           "  longer than that is cut off (no action), and with nothing left you wait until the\n"
           "  next round.\n")
_FREE = ("- Messages are free: after a successful `send_message` or `post_to_room` you act again "
         "in the same turn, up to {n} time(s) per turn. Any other call ends your turn.\n")

_REVIEWS = {"seller": "- Buyers can review their purchases; reviews and your average rating "
                      "are public.\n",
            "buyer": "- After a trade you can review it (`post_review`); reviews and each "
                     "seller's average rating are public.\n"}


def system_prompt(role: str, persona: str, situation: dict[str, Any], history: int,
                  quality: Any = None, overview: bool = True, board: bool = True,
                  buyer_values: str = "shared", swaps: bool = False,
                  excess: bool = False, free_messages: int = 0,
                  round_tokens: Optional[int] = None) -> str:
    """`quality`: the market's QualitySpec when some product's quality varies, else None.
    `board`: whether the public price board exists; `buyer_values`: shared | private."""
    rules = (_COMMON.format(role=role, history=history) + "\n"
             + _ROLE[role].format(**_WAYS[board], buyer_values=_VALUES[buyer_values]))
    if free_messages:
        rules += _FREE.format(n=free_messages)
    if round_tokens:
        rules += _BUDGET.format(n=round_tokens)
    if overview:
        rules += _OVERVIEW
    if swaps and role == "buyer":
        rules += "\n" + _SWAPS.format(grades=_GRADES if quality is not None else "",
                                      excess=_EXCESS if excess else "")
    if quality is not None:
        rules += "\n" + _QUALITY[role].format(
            certification=_CERT if quality.certification else "",
            reviews=_REVIEWS[role] if quality.reviews else "")
    return (f"{rules}\n{PERSONA_OPEN}\n{persona.strip()}\n{PERSONA_CLOSE}\n\n"
            f"YOUR SITUATION (fixed for this market; your own numbers are private):\n"
            f"{json.dumps(situation, indent=1, ensure_ascii=False)}\n")


def observation_message(obs: dict[str, Any], private_state: str,
                        last_result: dict[str, Any] | None) -> str:
    body: dict[str, Any] = {}
    if last_result is not None:
        body["last_action_result"] = last_result
    body["your_private_state"] = private_state
    body.update(obs)
    return json.dumps(body, ensure_ascii=False)
