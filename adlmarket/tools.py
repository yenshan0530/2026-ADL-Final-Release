"""The actions agents can take, as OpenAI function schemas.

One tool call per turn. Every tool takes an optional `private_state`: a note to the agent's
future self, shown back on its next turn. It is the only memory beyond the recent-turn
window, and nobody else ever sees it.
"""

from __future__ import annotations

import copy
import math
from typing import Any

_PRIVATE_STATE = {
    "type": "string",
    "description": "Optional note to your future self (plans, what you learned). Private.",
}

_PRICE = {"type": "number", "description": "US dollars per unit, at most two decimals."}
_QTY = {"type": "integer", "minimum": 1, "description": "Number of units."}
_MESSAGE = {"type": "string", "description": "What you say. Plain text."}


def _tool(name: str, description: str, properties: dict[str, Any],
          required: list[str]) -> dict[str, Any]:
    props = dict(properties)
    props["private_state"] = _PRIVATE_STATE
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required,
                           "additionalProperties": False},
        },
    }


TOOLS: dict[str, dict[str, Any]] = {
    "set_price": _tool(
        "set_price",
        "Seller. Post or change your public board price for one product. It goes on the board "
        "at the end of this turn; from then on any buyer may buy at it with `buy_listed`, "
        "while you have stock.",
        {"product_id": {"type": "string"}, "price": _PRICE},
        ["product_id", "price"]),
    "make_offer": _tool(
        "make_offer",
        "Seller. Make a BINDING private offer to one buyer. The buyer can accept it with "
        "`accept_offer` until it expires; you cannot withdraw it. Only offers and board "
        "prices are binding; a price written in a message is not.",
        {"buyer_id": {"type": "string"}, "product_id": {"type": "string"}, "price": _PRICE,
         "qty": _QTY, "message": {**_MESSAGE, "description": "Optional note sent with it."}},
        ["buyer_id", "product_id", "price", "qty"]),
    "buy_listed": _tool(
        "buy_listed",
        "Buyer. Buy at a seller's current public board price. Fills completely or not at all.",
        {"seller_id": {"type": "string"}, "product_id": {"type": "string"}, "qty": _QTY},
        ["seller_id", "product_id", "qty"]),
    "accept_offer": _tool(
        "accept_offer",
        "Buyer. Accept a binding offer a seller made to you, by its offer_id. Buys exactly "
        "the offered quantity at the offered price, or fails.",
        {"offer_id": {"type": "string"}},
        ["offer_id"]),
    "send_message": _tool(
        "send_message",
        "Send a private message to one other agent in your market (negotiate, ask, propose).",
        {"to": {"type": "string", "description": "An agent id."}, "message": _MESSAGE},
        ["to", "message"]),
    "post_to_room": _tool(
        "post_to_room",
        "Post to your side's group room: sellers see the seller room, buyers the buyer room.",
        {"message": _MESSAGE},
        ["message"]),
    "post_review": _tool(
        "post_review",
        "Buyer. Review one of your trades (after you know the grade you received). Every "
        "agent in the market sees the review and the seller's average rating.",
        {"trade_id": {"type": "string"},
         "rating": {"type": "integer", "minimum": 1, "maximum": 5,
                    "description": "1 (bad) to 5 (good)."},
         "text": {"type": "string", "description": "Optional comment."}},
        ["trade_id", "rating"]),
    "propose_swap": _tool(
        "propose_swap",
        "Buyer. Propose a BINDING trade to another buyer: units you hold and/or cash for "
        "units they hold. They can accept it with `accept_swap` until it expires. Cash alone "
        "is allowed.",
        {"buyer_id": {"type": "string", "description": "The other buyer."},
         "give": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0},
                  "description": "Units you hand over, {product_id: qty}. May be empty."},
         "get": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0},
                 "description": "Units you receive, {product_id: qty}. May be empty."},
         "cash": {"type": "number", "description": "Dollars that move with the swap: positive = "
                                                   "you pay them, negative = they pay you. "
                                                   "Default 0."},
         "give_grade": {"type": "object",
                        "additionalProperties": {"type": "string", "enum": ["high", "low"]},
                        "description": "Optional {product_id: grade}: which grade you hand over. "
                                       "Without it your low-grade units go first."},
         "get_grade": {"type": "object",
                       "additionalProperties": {"type": "string", "enum": ["high", "low"]},
                       "description": "Optional {product_id: grade}: which grade you ask for."},
         "message": {**_MESSAGE, "description": "Optional note sent with it."}},
        ["buyer_id", "give", "get"]),
    "accept_swap": _tool(
        "accept_swap",
        "Buyer. Accept a binding trade another buyer proposed to you, by its swap_id. "
        "Happens in full or fails (e.g. if either side no longer holds the units or cash).",
        {"swap_id": {"type": "string"}},
        ["swap_id"]),
    "wait": _tool(
        "wait",
        "Do nothing this turn (you can still update private_state).",
        {},
        []),
}

ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "seller": ("set_price", "make_offer", "send_message", "post_to_room", "wait"),
    "buyer": ("buy_listed", "accept_offer", "propose_swap", "accept_swap", "post_review",
              "send_message", "post_to_room", "wait"),
}

_GRADE = {"type": "string", "enum": ["high", "low"],
          "description": "For products whose quality varies: the grade you will deliver "
                         "(default high). Buyers do not see it before buying."}
_CERTIFIED = {"type": "boolean",
              "description": "Mark these units as certified high grade (shown to buyers). "
                             "Only allowed with grade high."}


def build_tools(role: str, spec: Any, has_quality: bool) -> list[dict[str, Any]]:
    """The tool schemas one role sees in one market: switched-off channels are left out,
    and the quality arguments and post_review appear only when the market uses them."""
    ch, q = spec.channels, spec.schema_.quality
    off = set()
    if not ch.board:
        off |= {"set_price", "buy_listed"}
    if not ch.dm:
        off.add("send_message")
    if not (ch.seller_room if role == "seller" else ch.buyer_room):
        off.add("post_to_room")
    if not (has_quality and q.reviews):
        off.add("post_review")
    if not getattr(spec, "buyer_swaps", False):
        off |= {"propose_swap", "accept_swap"}
    limit = getattr(spec, "max_message_chars", None)
    out = []
    for name in ROLE_TOOLS[role]:
        if name in off:
            continue
        tool = copy.deepcopy(TOOLS[name])
        msg = tool["function"]["parameters"]["properties"].get("message")
        if msg is not None and limit:
            # Tell the model the limit up front: a longer message is rejected and the
            # turn is lost.
            msg["maxLength"] = limit
            msg["description"] = f"{msg['description']} At most {limit} characters."
        if name in ("set_price", "make_offer") and has_quality:
            props = tool["function"]["parameters"]["properties"]
            props["grade"] = _GRADE
            if q.certification:
                props["certified"] = _CERTIFIED
            props["private_state"] = props.pop("private_state")     # keep it last
        if name == "propose_swap" and not has_quality:
            props = tool["function"]["parameters"]["properties"]
            props.pop("give_grade")
            props.pop("get_grade")
        out.append(tool)
    return out


def to_cents(value: Any) -> int:
    """Dollars (number or numeric string, at most two decimals) -> integer cents."""
    if isinstance(value, bool):
        raise ValueError("price must be a number")
    if isinstance(value, str):
        value = value.strip().lstrip("$").replace(",", "")
    amount = float(value)
    if not math.isfinite(amount):
        raise ValueError(f"price must be a finite number, got {value!r}")
    cents = round(amount * 100)
    if abs(amount * 100 - cents) > 1e-6:
        raise ValueError("price has more than two decimals")
    if cents <= 0:
        raise ValueError("price must be positive")
    return cents


def dollars(cents: int) -> float:
    return round(cents / 100, 2)
