"""What a run leaves for people to read, next to the machine-readable logs:

  markets/<id>/transcript.jsonl  every action of every agent (round, tick, agent, tool, args,
                                 result, its private_state note), always written; the full trajectory with model
                                 inputs and outputs is events.jsonl (save_trajectories)
  markets/<id>/transcript.md     the same market by round: messages, offers, board prices,
                                 trades, swaps, reviews, failed calls
  standings.json / standings.md  the final standings: per student (all its agents, e.g. one
                                 LoRA with three personas), per agent, per economy
  metadata.json                  when, how long, which command and commit, how many markets
                                 and model calls, tokens, failures, where the schemas came from
"""

from __future__ import annotations

import json
import platform
import statistics as st
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from adlmarket.tools import dollars

# ------------------------------------------------------------------- transcript ---

COMPACT_KEYS = ("round", "tick", "agent", "role", "tool", "args", "result", "free_action",
                "private_state")                  # the agent's note after this turn (<= 1000 chars)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    """A turn event without the observation and the model's input/output."""
    out = {k: row[k] for k in COMPACT_KEYS if k in row}
    llm = row.get("llm") or {}
    if llm.get("outcome") and llm["outcome"] != "accepted":
        out["model_outcome"] = llm["outcome"]
    return out


def _money(cents: Any) -> str:
    return f"${dollars(cents):,.2f}" if isinstance(cents, int) else str(cents)


def _md(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def transcript_md(market_id: str, rows: list[dict[str, Any]], final: dict[str, Any],
                  scores: list[dict[str, Any]], n_rounds: int) -> str:
    """One market, round by round, for people."""
    by_round: dict[int, list[str]] = defaultdict(list)
    events: list[tuple[int, int, str]] = []          # (round, tick, line)
    for m in final.get("messages", []):
        if m["channel"] == "notice":         # the engine's trade/expiry notices: shown as events
            continue
        where = {"dm": f"to {m['to']}", "seller_room": "in the seller room",
                 "buyer_room": "in the buyer room", "notice": f"notice to {m['to']}",
                 "review": "review"}.get(m["channel"], m["channel"])
        events.append((m["round"], m["tick"], f"**{m['sender']}** ({where}): {_md(m['text'])}"))
    for t in final.get("trades", []):
        grade = "" if t.get("grade", "high") == "high" else f", {t['grade']} grade"
        events.append((t["round"], t["tick"],
                       f"TRADE {t['id']}: {t['seller']} -> {t['buyer']}, {t['qty']} x {t['product']} "
                       f"at {_money(t['price'])} via {t['via']}{grade}"))
    for r in rows:
        res = r.get("result") or {}
        tool, args = r.get("tool"), r.get("args") or {}
        if tool == "make_offer" and res.get("ok"):
            events.append((r["round"], r["tick"],
                           f"offer by {r['agent']} to {args.get('buyer_id')}: {args.get('qty')} x "
                           f"{args.get('product_id')} at ${args.get('price')}"
                           + (f" ({res.get('offer_id')})" if res.get("offer_id") else "")))
        elif tool == "set_price" and res.get("ok"):
            events.append((r["round"], r["tick"], f"board: {r['agent']} lists {args.get('product_id')} "
                                                  f"at ${args.get('price')}"))
        elif tool in ("propose_swap", "accept_swap") and res.get("ok"):
            events.append((r["round"], r["tick"], f"swap: {r['agent']} {tool} {_md(json.dumps(args))}"))
        elif tool is None or not res.get("ok"):
            what = tool or "no call"
            events.append((r["round"], r["tick"],
                           f"_failed: {r['agent']} {what}: {_md(res.get('error', ''))}_"))
    for rnd, tick, line in sorted(events, key=lambda e: (e[0], e[1])):
        by_round[rnd].append(f"- t{tick} {line}")

    lines = [f"# Market {market_id}", ""]
    sellers = [s["id"] for s in final.get("sellers", [])]
    buyers = [b["id"] for b in final.get("buyers", [])]
    lines += [f"{len(sellers)} sellers: {', '.join(sellers)}", "",
              f"{len(buyers)} buyers: {', '.join(buyers)}", "",
              f"{n_rounds} rounds, {len(rows)} turns, {len(final.get('trades', []))} trades, "
              f"{len(final.get('messages', []))} messages", ""]
    for rnd in range(1, n_rounds + 1):
        lines += [f"## Round {rnd}", ""] + (by_round.get(rnd) or ["- (nothing happened)"]) + [""]
    lines += ["## Scores", "", "| agent | role | owner | raw | normalized | units |",
              "|---|---|---|---:|---:|---:|"]
    for s in sorted(scores, key=lambda s: (s["role"], -(s["normalized"] or -1e9))):
        norm = "-" if s["normalized"] is None else f"{s['normalized']:.3f}"
        lines.append(f"| {s['agent']} | {s['role']} | {s['owner']} | {s['raw']:.2f} | {norm} | "
                     f"{s.get('units_traded', 0)} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- statistics ---

def market_stats(final: dict[str, Any], turns: list[dict[str, Any]], n_rounds: int,
                 turns_per_round: int) -> dict[str, Any]:
    """How a market went: share of the possible gains from trade realized, share of the
    stock sold, and per round the units traded and the share of turns spent waiting or
    messaging; units traded in the final turn (a sign the market ran out of time)."""
    sellers, buyers = final["sellers"], final["buyers"]
    stock: dict[str, int] = defaultdict(int)
    cost: dict[str, int] = {}
    for s in sellers:
        for p, q in s["initial_stock"].items():
            stock[p] += q
        cost.update(s["cost"])
    values: dict[str, list[int]] = defaultdict(list)
    for b in buyers:
        for p, us in b["utility"].items():
            values[p] += [u for u in us if u > cost.get(p, 0)]
    possible = sum(sum(sorted(values[p], reverse=True)[:stock[p]]) - cost[p] * min(stock[p], len(values[p]))
                   for p in stock if p in cost)
    realized = sum(b["gained_utility"] for b in buyers) - sum(s["cost_of_sold"] for s in sellers)
    units = sum(t["qty"] for t in final["trades"])
    last_tick = n_rounds * turns_per_round
    per_round = []
    for r in range(1, n_rounds + 1):
        rows = [t for t in turns if t.get("round") == r]
        n = len(rows) or 1
        per_round.append({
            "round": r,
            "units": sum(t["qty"] for t in final["trades"] if t["round"] == r),
            "wait_share": round(sum(1 for t in rows if t.get("tool") in ("wait", None)) / n, 3),
            "message_share": round(sum(1 for t in rows if t.get("tool") in ("send_message", "post_to_room")) / n, 3)})
    return {"possible_gains": round(possible / 100, 2), "realized_gains": round(realized / 100, 2),
            "efficiency": round(realized / possible, 4) if possible > 0 else None,
            "stock": sum(stock.values()), "units_sold": units,
            "sold_share": round(units / sum(stock.values()), 4) if stock else None,
            "final_turn_units": sum(t["qty"] for t in final["trades"] if t["tick"] == last_tick),
            "per_round": per_round}


def economy_stats(per_market: dict[str, dict[str, Any]], repeat_of: dict[str, int]) -> dict[str, Any]:
    """Means over the markets of each economy, and over all markets."""
    def summary(ms: list[dict[str, Any]]) -> dict[str, Any]:
        effs = [m["efficiency"] for m in ms if m["efficiency"] is not None]
        rounds = max((len(m["per_round"]) for m in ms), default=0)
        units = sum(m["units_sold"] for m in ms) or 1
        return {"markets": len(ms),
                "efficiency": _mean(effs), "efficiency_min": round(min(effs), 4) if effs else None,
                "sold_share": _mean([m["sold_share"] for m in ms if m["sold_share"] is not None]),
                "final_turn_unit_share": round(sum(m["final_turn_units"] for m in ms) / units, 4),
                "per_round": [{"round": r + 1,
                               "unit_share": round(sum(m["per_round"][r]["units"] for m in ms
                                                       if len(m["per_round"]) > r) / units, 4),
                               "wait_share": _mean([m["per_round"][r]["wait_share"] for m in ms
                                                    if len(m["per_round"]) > r]),
                               "message_share": _mean([m["per_round"][r]["message_share"] for m in ms
                                                       if len(m["per_round"]) > r])}
                              for r in range(rounds)]}
    by_e: dict[int, list] = defaultdict(list)
    for mid, m in per_market.items():
        by_e[repeat_of[mid]].append(m)
    return {"all": summary(list(per_market.values())),
            "economies": {str(e): summary(ms) for e, ms in sorted(by_e.items())}}


def stats_lines(stats: dict[str, Any]) -> list[str]:
    a = stats["all"]
    f = lambda x: "-" if x is None else f"{x:.0%}"
    out = [f"market health ({a['markets']} markets): {f(a['efficiency'])} of the possible gains "
           f"realized (lowest market {f(a['efficiency_min'])}), {f(a['sold_share'])} of the stock "
           f"sold, {f(a['final_turn_unit_share'])} of the units in the final turn"]
    out.append("  per round: " + "; ".join(
        f"r{r['round']} {f(r['unit_share'])} of units, {f(r['wait_share'])} waits" for r in a["per_round"]))
    for e, s in stats["economies"].items():
        out.append(f"  economy {e}: {f(s['efficiency'])} realized, {f(s['sold_share'])} sold")
    return out


# -------------------------------------------------------------------- standings ---

def _mean(xs: list[float]) -> Optional[float]:
    return round(st.mean(xs), 4) if xs else None


def _se(xs: list[float]) -> Optional[float]:
    return round(st.stdev(xs) / len(xs) ** 0.5, 4) if len(xs) > 1 else None


def _ta_base(agent: str, ta_ids: set[str]) -> str:
    """A TA copy (`ta_buyer_busy_2`, made when a market needs more TAs than the pool has)
    counts as its TA."""
    head, _, tail = agent.rpartition("_")
    return head if tail.isdigit() and head in ta_ids else agent


def anchor_value(values: list[float], percentile: float) -> tuple[int, float]:
    """(index, value) of the anchor at `percentile` among `values` sorted ascending:
    position floor(p/100 x (n - 1)), i.e. with few values the lower one (favors students)."""
    order = sorted(values)
    i = int(percentile / 100 * (len(order) - 1) + 1e-9)
    return i, order[i]


def grade(x: float, lo: float, hi: float, g: Any) -> Optional[float]:
    """The straight line through (lo, low_score) and (hi, high_score) at x; not capped."""
    if hi == lo:
        return None
    return round(g.low_score + (x - lo) * (g.high_score - g.low_score) / (hi - lo), 4)


def standings(rows: list[dict[str, Any]], grading: Any = None, graded_roles: tuple = (),
              rounds: Optional[int] = None) -> dict[str, Any]:
    """Standings from the score rows: per student and role (all its agents, e.g. one LoRA
    with several personas), per agent, per economy. The ranking metric is the raw score
    (seller profit, buyer surplus, in dollars); with `grading`, students of the graded
    roles also get a grade from the TA anchors of each economy (GradingSpec), and are
    ranked by it. Nothing is capped."""
    def stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        norms = [r["normalized"] for r in rs if r["normalized"] is not None]
        raws = [r["raw"] for r in rs]
        out = {"markets": len(rs), "mean_raw": round(st.mean(raws), 2), "se_raw": _se(raws),
               "mean_normalized": _mean(norms),
               "units": sum(r.get("units_traded", 0) for r in rs),
               "model_failures": sum(r.get("model_failures", 0) for r in rs),
               "no_call_turns": sum(r.get("no_call_turns", 0) for r in rs)}
        if rounds:
            out["mean_raw_per_round"] = round(st.mean(raws) / rounds, 2)
        return out

    def rank(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        val = lambda r: r.get(key) if r.get(key) is not None else -1e18
        items.sort(key=lambda r: (r["role"], -val(r)))
        pos: dict[str, int] = defaultdict(int)
        for r in items:
            pos[r["role"]] += 1
            r["rank"] = pos[r["role"]]
        return items

    ta_ids = {r["agent"] for r in rows if r["owner"] == "ta"}
    by_owner, by_agent, by_owner_econ = defaultdict(list), defaultdict(list), defaultdict(list)
    ta_econ: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_agent[(r["agent"], r["role"], r["owner"])].append(r)
        e = r.get("repeat", 0)
        if r["owner"] != "ta":
            by_owner[(r["owner"], r["role"])].append(r)
            by_owner_econ[(r["owner"], r["role"], e)].append(r)
        elif r["role"] in graded_roles:
            ta_econ[(r["role"], e)][_ta_base(r["agent"], ta_ids)].append(r["raw"])

    anchors: dict[str, dict[str, Any]] = defaultdict(dict)
    if grading is not None:
        for (role, e), per_ta in sorted(ta_econ.items()):
            names = sorted(per_ta, key=lambda a: st.mean(per_ta[a]))
            means = [st.mean(per_ta[a]) for a in names]
            i_lo, lo = anchor_value(means, grading.low_percentile)
            i_hi, hi = anchor_value(means, grading.high_percentile)
            anchors[role][str(e)] = {
                "tas": len(names),
                "low": {"percentile": grading.low_percentile, "score": grading.low_score,
                        "agent": names[i_lo], "raw": round(lo, 2)},
                "high": {"percentile": grading.high_percentile, "score": grading.high_score,
                         "agent": names[i_hi], "raw": round(hi, 2)}}

    students = []
    for (o, role), rs in by_owner.items():
        row = {"owner": o, "role": role, "agents": len({r["agent"] for r in rs}), **stats(rs)}
        econs = sorted({r.get("repeat", 0) for r in rs})
        row["by_economy"] = {str(e): round(st.mean(r["raw"] for r in by_owner_econ[(o, role, e)]), 2)
                             for e in econs}
        if grading is not None and role in graded_roles:
            per = {}
            for e in econs:
                a = anchors[role].get(str(e))
                if a:
                    per[str(e)] = grade(row["by_economy"][str(e)], a["low"]["raw"], a["high"]["raw"], grading)
            row["grade_by_economy"] = per
            vals = [v for v in per.values() if v is not None]
            row["grade"] = round(st.mean(vals), 4) if vals else None
        students.append(row)
    graded = grading is not None
    students = rank(students, "grade" if graded else "mean_raw")
    agents = rank([{"agent": a, "role": role, "owner": o, **stats(rs)}
                   for (a, role, o), rs in by_agent.items()], "mean_raw")
    return {"metric": "raw: seller profit (revenue - cost), buyer surplus (value - paid), dollars",
            "grading": grading.model_dump() if graded else None,
            "anchors": dict(anchors), "students": students, "agents": agents}


def standings_md(stand: dict[str, Any], title: str, mock: bool = False) -> str:
    f = lambda x: "-" if x is None else f"{x:.3f}"
    m = lambda x: "-" if x is None else f"{x:.2f}"
    g = stand.get("grading")
    lines = [f"# Standings: {title}", "",
             "Raw score: seller profit (revenue - cost of units sold), buyer surplus (value of "
             "units bought - money paid), in dollars per market; a student's row pools all of "
             "its agents (one per persona). se = standard error over markets. failed calls = "
             "turns with no valid tool call (e.g. a reply cut off at the token limit) or no "
             "answer from the model server."]
    if mock:
        lines += ["", "**MOCK RUN** (--mock-llm): the LLM agents were replaced by rule-based "
                      "stand-ins. These are not results of the real agents."]
    if g:
        lines += ["", f"Grade: per economy, the TA agents of the graded role are ranked by raw "
                      f"score; the TA at percentile {g['low_percentile']:g} gets {g['low_score']:g}, "
                      f"the one at {g['high_percentile']:g} gets {g['high_score']:g} (with few TAs "
                      f"the lower one); a student's grade is the line through the two at its raw "
                      f"score (not capped), averaged over economies."]
    lines.append("")
    for role, per_e in (stand.get("anchors") or {}).items():
        lines += [f"## Grade anchors ({role}s)", "", "| economy | TAs | low anchor | raw | high anchor | raw |",
                  "|---:|---:|---|---:|---|---:|"]
        for e, a in per_e.items():
            lines.append(f"| {e} | {a['tas']} | {a['low']['agent']} | {a['low']['raw']:.2f} | "
                         f"{a['high']['agent']} | {a['high']['raw']:.2f} |")
        lines.append("")
    for role in ("seller", "buyer"):
        rows = [s for s in stand["students"] if s["role"] == role]
        if not rows:
            continue
        econs = sorted({e for s in rows for e in s["by_economy"]}, key=int)
        has_grade = any("grade" in s for s in rows)
        head = "| rank | student | agents | markets | " + ("grade | " if has_grade else "") + \
               "mean raw $ | se | per round $ | normalized | " + " | ".join(f"econ {e} raw" for e in econs) + \
               (" | " + " | ".join(f"econ {e} grade" for e in econs) if has_grade else "") + " |"
        lines += [f"## Student {role}s", "", head,
                  "|" + "---:|" * (head.count("|") - 1)]
        for s in rows:
            cells = [str(s["rank"]), s["owner"], str(s["agents"]), str(s["markets"])]
            if has_grade:
                cells.append(f(s.get("grade")))
            cells += [m(s["mean_raw"]), m(s["se_raw"]), m(s.get("mean_raw_per_round")),
                      f(s["mean_normalized"])]
            cells += [m(s["by_economy"].get(e)) for e in econs]
            if has_grade:
                cells += [f((s.get("grade_by_economy") or {}).get(e)) for e in econs]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    for role in ("seller", "buyer"):
        rows = [a for a in stand["agents"] if a["role"] == role]
        if not rows:
            continue
        lines += [f"## All {role} agents", "",
                  "| rank | agent | owner | markets | mean raw $ | se | normalized | failed calls |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"]
        for a in rows:
            lines.append(f"| {a['rank']} | {a['agent']} | {a['owner']} | {a['markets']} | "
                         f"{m(a['mean_raw'])} | {m(a['se_raw'])} | {f(a['mean_normalized'])} | "
                         f"{a['model_failures'] + a['no_call_turns']} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- metadata ---

def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def inference_totals(path: Path) -> dict[str, Any]:
    """Model calls by outcome and token use, from inference.jsonl."""
    out: dict[str, Any] = {"calls": 0, "outcomes": defaultdict(int), "input_tokens": 0,
                           "output_tokens": 0, "reasoning_tokens": 0}
    if not path.exists():
        return {"calls": 0}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out["calls"] += 1
        out["outcomes"][r.get("outcome") or "none"] += 1
        u = r.get("usage") or {}               # the client's normalized usage (client.py)
        for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
            out[k] += u.get(k) or 0
    out["outcomes"] = dict(out["outcomes"])
    return out


def metadata(runner: Any, plans: list, started: float, scores: list[dict[str, Any]]) -> dict[str, Any]:
    run, cfg = runner.run, runner.cfg
    root = run.root
    return {
        "run": runner.out.name,
        "status": "finished",
        "script": (cfg.origin or {}).get("script"),
        "phase": (cfg.origin or {}).get("phase"),
        "command": runner.invocation,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(time.time() - started, 1),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "seed": cfg.seed,
        "mock_llm": runner.mock_llm,
        "economies": cfg.casting.repeats,
        "markets": len(plans),
        "rotate": cfg.casting.rotate,
        "agents": {"sellers": len(run.sellers), "buyers": len(run.buyers),
                   "padding": {k: len(v) for k, v in run.pads.items()}},
        "items": {str(r): f for r, f in sorted(runner.schema_files.items())},
        "turns": sum(1 for _ in _iter_turns(runner.out)),
        "trades": sum(_count(runner.out / "markets" / p.market_id / "final.json", "trades") for p in plans),
        "messages": sum(_count(runner.out / "markets" / p.market_id / "final.json", "messages")
                        for p in plans),
        "score_rows": len(scores),
        "model_calls": inference_totals(runner.out / "inference.jsonl"),
    }


def _count(path: Path, key: str) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get(key, []))
    except (OSError, ValueError):
        return 0


def _iter_turns(out: Path):
    for f in (out / "markets").glob("*/transcript.jsonl"):
        yield from f.open(encoding="utf-8")
