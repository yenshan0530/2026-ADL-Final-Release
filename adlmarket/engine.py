"""Run loop: plan markets from the casting, play each one, write logs and scores.

Within a market, time is rounds x turns. On each turn every agent sees the same snapshot
and decides at the same time (LLM calls run in parallel); the actions are then applied in
a fresh random order, so when two buyers race for the last unit the winner is random,
not whoever happens to be listed first.
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from adlmarket.casting import MarketSeats, plan_markets
from adlmarket.config import AgentSpec, LoadedRun
from adlmarket.market import Market, load_catalog
from adlmarket.policies import MIN_CALL_TOKENS, MOCK_STRATEGY, SCRIPTED, Decision, LLMPolicy, Policy
from adlmarket.prompts import system_prompt
from adlmarket.scoring import score_market
from adlmarket.report import (compact, economy_stats, market_stats, metadata, standings,
                              standings_md, transcript_md)
from adlmarket.schema import Schema, build_items_schema


_WRITE_LOCK = threading.Lock()         # markets running in parallel share run-level files


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def with_reasoning(model: Any, on: bool) -> Any:
    """The model spec with the agent's thinking mode, for local open-weight models (sent as
    chat_template_kwargs.enable_thinking; Qwen3.5's chat template reads it). Hosted APIs
    are left as configured (they use reasoning_effort)."""
    if model.api_key or model.api_key_env:
        return model
    extra = dict(model.extra_body or {})
    kwargs = dict(extra.get("chat_template_kwargs") or {})
    if kwargs.get("enable_thinking") == on:
        return model
    kwargs["enable_thinking"] = on
    return model.model_copy(update={"extra_body": {**extra, "chat_template_kwargs": kwargs}})


class Runner:
    def __init__(self, run: LoadedRun, *, mock_llm: bool = False, client: Any = None,
                 out_dir: Optional[Path] = None, invocation: Optional[str] = None,
                 resume: bool = False):
        self.run = run
        self.cfg = run.config
        self.mock_llm = mock_llm
        self.catalog = load_catalog(run.root / self.cfg.market.catalog)
        self.catalog_by_id = {p.id: p for p in self.catalog}
        self.schemas: dict[int, Schema] = {}
        if client is None and not mock_llm and self._needs_llm():
            from adlmarket.client import ModelClient
            client = ModelClient(api_key="EMPTY", base_url="http://localhost:8000/v1",
                                 timeout=self.cfg.call_timeout, retry_limit=self.cfg.call_retries,
                                 min_interval=self.cfg.min_call_interval)
        self.client = client
        self.invocation = invocation
        self.out = out_dir or self._new_out_dir()
        self.schema_files: dict[int, str] = {}       # economy -> its item file
        self.resume = resume
        if resume and not (self.out / "config.yaml").exists():
            raise FileNotFoundError(f"{self.out} is not a run folder (no config.yaml)")
        # Checked before anything (e.g. vLLM logs) is written into the folder.
        if not resume and self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"output folder {self.out} is not empty; pick another one "
                                  f"so this run's logs and scores are not mixed with older ones")

    def _new_out_dir(self) -> Path:
        """<output_dir>/<script>_<timestamp>, e.g. results/run_economy_20261108-153000
        (a suffix -2, -3 ... if that folder already exists)."""
        script = Path((self.cfg.origin or {}).get("script") or self.cfg.name).stem
        base = self.run.root / self.cfg.output_dir / f"{script}_{time.strftime('%Y%m%d-%H%M%S')}"
        out, k = base, 2
        while out.exists():
            out, k = base.with_name(f"{base.name}-{k}"), k + 1
        return out

    def _needs_llm(self) -> bool:
        return any(a.controller == "llm" for a in self.run.everyone())

    # ------------------------------------------------------------------ policies ---

    def _policy(self, agent: AgentSpec, market: Market) -> tuple[Policy, dict[str, Any]]:
        spec = self.cfg.market
        situation = market.situation(agent.id)
        if agent.controller == "scripted" or self.mock_llm:
            strategy = agent.strategy if agent.controller == "scripted" else MOCK_STRATEGY[agent.role]
            if strategy not in SCRIPTED:
                raise ValueError(f"agent {agent.id}: unknown strategy {strategy!r}; "
                                 f"known: {sorted(SCRIPTED)}")
            return SCRIPTED[strategy](agent, situation, spec), {"strategy": strategy}
        system = system_prompt(agent.role, agent.persona_body(), situation, spec.history_turns,
                               quality=spec.schema_.quality if market.has_quality else None,
                               overview=spec.market_overview != "none",
                               board=spec.channels.board, buyer_values=spec.buyer_values,
                               swaps=spec.buyer_swaps, excess=spec.scoring.allow_excess_purchase,
                               free_messages=spec.rounds.free_messages,
                               round_tokens=spec.rounds.output_tokens_per_round)
        model = with_reasoning(self.run.models[agent.model], agent.reasoning)
        policy = LLMPolicy(agent, model, self.client, system, market.tool_schemas(agent.role),
                           spec.history_turns, spec.private_state_chars, seed=self.cfg.seed,
                           save_prompts=self.cfg.save_prompts and self.cfg.save_trajectories,
                           history_board=spec.history_board)
        meta = {"model": agent.model, "system_prompt": system}
        if self.cfg.save_prompts:
            meta["tools"] = market.tool_schemas(agent.role)
        return policy, meta

    # ---------------------------------------------------------------------- market ---

    def play(self, seats: MarketSeats) -> list[dict[str, Any]]:
        spec = self.cfg.market
        rng = random.Random(f"market:{self.cfg.seed}:{seats.seed_id}")
        market = Market.generate(seats.market_id, spec, self.schema_for(seats.repeat),
                                 self.catalog_by_id, seats.sellers, seats.buyers, rng,
                                 market_index=seats.index, rotation=seats.rotation,
                                 seed_id=seats.seed_id)
        mdir = self.out / "markets" / seats.market_id
        mdir.mkdir(parents=True, exist_ok=True)
        trace = self.cfg.save_trajectories
        events = mdir / "events.jsonl"
        turns: list[dict[str, Any]] = []         # every action, always (transcript.jsonl)

        def log_event(row: dict[str, Any]) -> None:
            if row.get("type") == "turn":
                turns.append(compact(row))
            if trace:
                _jsonl(events, row)

        agents = {a.id: a for a in seats.sellers + seats.buyers}
        policies: dict[str, Policy] = {}
        for a in agents.values():
            policies[a.id], meta = self._policy(a, market)
            if not trace:
                meta.pop("system_prompt", None)
            _jsonl(mdir / "agents.jsonl", {"agent": a.id, "role": a.role, "owner": a.owner,
                                            "controller": "mock" if self.mock_llm and a.controller == "llm"
                                            else a.controller, "source": a.source,
                                            "situation": market.situation(a.id), **meta})
        n_rounds = spec.rounds.total
        log_event({"type": "market_start", "market": seats.market_id, "rounds": n_rounds,
                        "initial": market.snapshot()})

        failed = {a: 0 for a in agents}           # model calls failed in a row
        counts = {a: {"model_failures": 0, "no_call_turns": 0} for a in agents}
        T = spec.rounds.turns_per_round
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.max_workers)) as pool:
            limit = spec.rounds.output_tokens_per_round
            for r in range(1, n_rounds + 1):
                market.round = r
                # rounds.output_tokens_per_round: each LLM agent's budget, renewed every round
                budget = ({a: limit for a, p in policies.items() if isinstance(p, LLMPolicy)}
                          if limit else {})
                for t in range(T):
                    market.tick += 1
                    actors = sorted(agents)
                    for extra in range(spec.rounds.free_messages + 1):
                        actors = self._sub_step(market, agents, actors, policies, pool, rng, failed,
                                                counts, seats.market_id, r, T - t, extra, log_event,
                                                budget)
                        if not actors:
                            break
                    market.end_turn()                # board prices set this turn go up now
                market.end_round()
                log_event({"type": "round_end", "round": r})

        scores = score_market(market)
        for row in scores:
            row.update({"market": seats.market_id, "repeat": seats.repeat,
                        "owner": agents[row["agent"]].owner, **counts[row["agent"]]})
        final = market.snapshot()                # messages included: they are the discussion
        (mdir / "final.json").write_text(json.dumps(final, indent=1), encoding="utf-8")
        log_event({"type": "market_end", "scores": scores})
        with (mdir / "transcript.jsonl").open("w", encoding="utf-8") as fh:
            for row in turns:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (mdir / "transcript.md").write_text(
            transcript_md(seats.market_id, turns, final, scores, n_rounds), encoding="utf-8")
        (mdir / "stats.json").write_text(json.dumps(
            market_stats(final, turns, n_rounds, spec.rounds.turns_per_round), indent=1),
            encoding="utf-8")
        return scores

    def _sub_step(self, market: Market, agents: dict, actors: list[str], policies: dict[str, Policy],
                  pool: ThreadPoolExecutor, rng: random.Random, failed: dict[str, int], counts: dict,
                  market_id: str, r: int, turns_left: int, extra: int, log_event,
                  budget: Optional[dict[str, int]] = None) -> list[str]:
        """`actors` decide at once, then act in random order. extra > 0: a free action after a
        message (rounds.free_messages). Returns who sent a message successfully (may act again)."""
        order = sorted(actors)
        obs = {a: market.observe(a, turns_left=turns_left) for a in order}
        if extra:
            left = market.spec.rounds.free_messages - extra
            for o in obs.values():
                o["free_action"] = (f"Your message did not use your turn: act again "
                                    f"({left} more free message(s) this turn).")
        budget = budget if budget is not None else {}
        spent_out = {a for a in order if a in budget and budget[a] < MIN_CALL_TOKENS}
        for a in order:
            if a in budget:
                obs[a]["output_tokens_left_this_round"] = budget[a]
                policies[a].budget = budget[a]
        futures = {a: pool.submit(self._decide, policies[a], obs[a]) for a in order
                   if a not in spent_out}
        decisions = {a: f.result() for a, f in futures.items()}
        for a in spent_out:                      # the round's output budget is used up: wait
            decisions[a] = Decision("wait", {}, private_state=getattr(policies[a], "private_state", ""),
                                    log={"outcome": "round_budget_used", "output_tokens_spent": 0})
        for a, d in decisions.items():
            if a in budget and d.log:
                budget[a] = max(0, budget[a] - int(d.log.get("output_tokens_spent") or 0))
        rng.shuffle(order)
        for a in sorted(actors):
            d = decisions[a]
            if d.log:
                self._log_call(market_id, r, market.tick, a, d)
            failed[a] = failed[a] + 1 if d.infra else 0
            if failed[a] >= self.cfg.max_failed_calls_in_a_row:
                raise RuntimeError(
                    f"market {market_id}: {failed[a]} model calls in a row "
                    f"failed for agent {a} ({d.log.get('model')} at "
                    f"{d.log.get('base_url')}): {d.log.get('error')}. Run stopped; "
                    f"see {self.out / 'inference.jsonl'}")
        again = []
        for a in order:
            d = decisions[a]
            if d.tool is None:
                result = d.error or {"ok": False, "error": "no_tool_call",
                                     "detail": "no function call this turn"}
                counts[a]["model_failures" if d.infra else "no_call_turns"] += 1
            else:
                result = market.apply(a, d.tool, d.args).as_dict()
            policies[a].observe_result(result)
            row = {"type": "turn", "round": r, "tick": market.tick,
                   "agent": a, "role": agents[a].role, "observation": obs[a],
                   "tool": d.tool, "args": d.args, "result": result,
                   "private_state": d.private_state, "llm": d.log or None}
            if extra:
                row["free_action"] = extra
            log_event(row)
            if d.tool in ("send_message", "post_to_room") and result.get("ok"):
                again.append(a)
        return again

    def schema_for(self, repeat: int) -> Schema:
        """One schema per economy (repeat), shared by all of its markets."""
        return self.schemas[repeat]

    def prepare_schemas(self, repeats: int, plans: Optional[list[MarketSeats]] = None) -> None:
        """Each economy's items, kinds and prices, from its item file (nothing is random)."""
        self.cfg.check_items()
        spec = self.cfg.market.schema_
        for r in range(repeats):
            sizes = {len(p.sellers) for p in plans or [] if p.repeat == r}
            if len(sizes) > 1:
                raise ValueError(f"items are dealt to one bundle per seller seat, so every market "
                                 f"needs the same number of sellers; economy {r} has markets of "
                                 f"{sorted(sizes)}")
            n_sellers = sizes.pop() if sizes else self.cfg.casting.sellers.per_market
            items_file = spec.items_file(r)
            self.schemas[r] = build_items_schema(f"{self.out.name}_economy_{r}", spec, items_file,
                                                 self.catalog, self.run.root, n_sellers)
            self.schema_files[r] = items_file
        files = sorted(set(self.schema_files.values()))
        print(f"items: {', '.join(files)} ({len(self.schemas[0].plans)} items per economy"
              f"{'' if len(files) > 1 else ', every economy'})", flush=True)

    def write_configs(self) -> None:
        """<output>/config.yaml (the config as played) and a first metadata.json (script,
        phase, command), so --resume can rebuild the run; see snapshot.write_run_config."""
        from adlmarket.snapshot import write_run_config
        write_run_config(self)
        origin = self.cfg.origin or {}
        start = {"run": self.out.name, "status": "running", "script": origin.get("script"),
                 "phase": origin.get("phase"), "command": self.invocation,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mock_llm": self.mock_llm}
        (self.out / "metadata.json").write_text(json.dumps(start, indent=1), encoding="utf-8")

    def _log_call(self, market_id: str, round_: int, tick: int, agent: str, d: Decision) -> None:
        """One line per model call, always written (also without save_trajectories): the
        run's record of endpoint health, truncation, retries and token use."""
        keep = ("model", "base_url", "outcome", "error", "stop_reason", "usage", "timing",
                "attempts_infra", "parse_error", "arguments_repaired", "parsed_from_text",
                "history_turns_shown", "length_retries", "truncated_call",
                "output_tokens_spent", "max_tokens", "context_retries")
        _jsonl(self.out / "inference.jsonl", {
            "market": market_id, "round": round_, "tick": tick, "agent": agent, "tool": d.tool,
            "result_error": (d.error or {}).get("error") if d.tool is None else None,
            **{k: d.log[k] for k in keep if d.log.get(k) is not None}})

    @staticmethod
    def _decide(policy: Policy, obs: dict[str, Any]) -> Decision:
        try:
            return policy.decide(obs)
        except Exception as exc:          # a harness crash skips the turn, not the run
            return Decision(None, {}, log={"outcome": "harness_error",
                                           "error": f"{type(exc).__name__}: {exc}"[:500]},
                            infra=True, error={"ok": False, "error": "model_call_failed",
                                               "detail": "the harness failed to run your agent "
                                                         "this turn (not counted against you)"})

    # ------------------------------------------------------------------------ run ---

    def run_all(self, only: Optional[int] = None) -> dict[str, Any]:
        plans = plan_markets(self.cfg.casting, self.run.sellers, self.run.buyers, self.cfg.seed,
                             pads=self.run.pads)
        if only is not None:
            plans = plans[:only]
        started = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        self.prepare_schemas(self.cfg.casting.repeats, plans)
        plan_rows = [{"market": p.market_id, "repeat": p.repeat, "sellers": [a.id for a in p.sellers],
                      "buyers": [a.id for a in p.buyers]} for p in plans]
        done: dict[str, list[dict[str, Any]]] = {}       # market -> its score rows
        if self.resume:
            done = self._resume_state(plans, plan_rows)
        else:
            self.write_configs()
            (self.out / "plan.json").write_text(json.dumps(plan_rows, indent=1), encoding="utf-8")
        todo = [p for p in plans if p.market_id not in done]
        lock = threading.Lock()

        def play_and_record(p: MarketSeats) -> list[dict[str, Any]]:
            rows = self.play(p)
            # markets/<id>/scores.json marks the market as finished (written atomically):
            # --resume skips markets that have it and replays the others
            _atomic_write(self.out / "markets" / p.market_id / "scores.json", json.dumps(rows, indent=1))
            for row in rows:
                _jsonl(self.out / "scores.jsonl", row)
            with lock:                                   # standings as of every finished market
                done[p.market_id] = rows
                self._write_standings([r for q in plans if q.market_id in done for r in done[q.market_id]])
                in_economy = [q for q in plans if q.repeat == p.repeat]
                print(f"market {len(done)}/{len(plans)} done: {p.market_id} (economy {p.repeat}: "
                      f"{sum(q.market_id in done for q in in_economy)}/{len(in_economy)}; "
                      f"{(time.time() - started) / 60:.1f} min so far); standings: "
                      f"{_rel(self.out / 'standings.md', self.run.root)}", flush=True)
            return rows

        workers = max(1, min(self.cfg.parallel_markets, len(todo) or 1))
        if workers == 1:
            for p in todo:
                play_and_record(p)
        else:
            with ThreadPoolExecutor(max_workers=workers) as markets:
                futures = [markets.submit(play_and_record, p) for p in todo]
                try:
                    for f in futures:                   # an error stops the run
                        f.result()
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
        all_scores = [r for p in plans for r in done.get(p.market_id, [])]     # plan order
        summary = summarize(all_scores)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        by_owner = summarize_owners(all_scores)
        stand = self._write_standings(all_scores)
        per_market = {}
        for p in plans:
            f = self.out / "markets" / p.market_id / "stats.json"
            if f.exists():
                per_market[p.market_id] = json.loads(f.read_text(encoding="utf-8"))
        stats = economy_stats(per_market, {p.market_id: p.repeat for p in plans})
        (self.out / "economy_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        meta = metadata(self, plans, started, all_scores)
        (self.out / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return {"out": str(self.out), "markets": len(plans), "summary": summary,
                "by_owner": by_owner, "standings": stand, "metadata": meta, "stats": stats}

    def _write_standings(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        stand = standings(rows, self.cfg.grading, tuple(self.cfg.graded_roles),
                          rounds=self.cfg.market.rounds.total)
        _atomic_write(self.out / "standings.json", json.dumps(stand, indent=1))
        _atomic_write(self.out / "standings.md", standings_md(stand, self.out.name, self.mock_llm))
        return stand

    def _resume_state(self, plans: list[MarketSeats], plan_rows: list[dict[str, Any]]
                      ) -> dict[str, list[dict[str, Any]]]:
        """Continue a stopped run: check it is the same run, keep its finished markets,
        delete half-played ones, and rebuild scores.jsonl from the finished markets."""
        saved = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        if saved != plan_rows:
            raise ValueError(f"{self.out}: the seating differs from the run's plan.json (agents, "
                             f"personas or settings changed since it started); not resuming")
        done: dict[str, list[dict[str, Any]]] = {}
        replayed = []
        for p in plans:
            mdir = self.out / "markets" / p.market_id
            if (mdir / "scores.json").exists():
                done[p.market_id] = json.loads((mdir / "scores.json").read_text(encoding="utf-8"))
            elif mdir.exists():
                shutil.rmtree(mdir)                      # half-played: play it again
                replayed.append(p.market_id)
        with (self.out / "scores.jsonl").open("w", encoding="utf-8") as fh:
            for p in plans:
                for row in done.get(p.market_id, []):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        entry = {"resumed": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": self.invocation,
                 "finished_before": len(done), "half_played_restarted": replayed,
                 "to_play": len(plans) - len(done), "mock_llm": self.mock_llm}
        _jsonl(self.out / "resumes.jsonl", entry)
        print(f"resuming {self.out.name}: {len(done)} of {len(plans)} markets finished, "
              f"{len(replayed)} half-played restarted, {len(plans) - len(done)} to play", flush=True)
        return done


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean score per agent across all its markets, best first within each role."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)
    out = []
    for agent, rs in by_agent.items():
        norms = [r["normalized"] for r in rs if r["normalized"] is not None]
        out.append({"agent": agent, "role": rs[0]["role"], "owner": rs[0]["owner"],
                    "markets": len(rs),
                    "mean_normalized": round(sum(norms) / len(norms), 4) if norms else None,
                    "mean_raw": round(sum(r["raw"] for r in rs) / len(rs), 2),
                    "model_failures": sum(r.get("model_failures", 0) for r in rs),
                    "no_call_turns": sum(r.get("no_call_turns", 0) for r in rs)})
    out.sort(key=lambda r: (r["role"], -(r["mean_normalized"] or -1e9)))
    return out


def summarize_owners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean score per student (owner) and role over all markets of all its agents, e.g.
    one LoRA played with three personas. TA agents are left out."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r["owner"] != "ta":
            by.setdefault((r["owner"], r["role"]), []).append(r)
    out = []
    for (owner, role), rs in by.items():
        norms = [r["normalized"] for r in rs if r["normalized"] is not None]
        out.append({"owner": owner, "role": role, "agents": len({r["agent"] for r in rs}),
                    "markets": len(rs),
                    "mean_normalized": round(sum(norms) / len(norms), 4) if norms else None,
                    "mean_raw": round(sum(r["raw"] for r in rs) / len(rs), 2)})
    out.sort(key=lambda r: (r["role"], -(r["mean_normalized"] or -1e9)))
    return out
