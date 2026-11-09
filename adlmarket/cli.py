"""Shared command-line plumbing for run_economy.py and run_market.py (repo root)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from contextlib import ExitStack

from adlmarket.casting import plan_markets
from adlmarket.config import LoadedRun, check_credentials
from adlmarket.serving import VLLMCluster, register_adapters


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """The flags both scripts take. Every setting of a run is in its config file; these
    only say what to do with it."""
    ap.add_argument("--dry-run", action="store_true",
                    help="check the config and print who plays in which market; play nothing")
    ap.add_argument("--mock-llm", action="store_true",
                    help="replace every LLM agent with a scripted one (no GPU/API needed)")
    ap.add_argument("--launch-vllm", action="store_true",
                    help="start the vLLM server(s) of configs/models.yaml, run, then stop them "
                         "(otherwise one must be running already, e.g. from run_local.sh)")
    ap.add_argument("--resume", metavar="RUN_DIR",
                    help="continue a stopped run (results/<run>): its config.yaml is played "
                         "again; finished markets are kept, half-played ones replayed")


def describe(run: LoadedRun) -> None:
    print(f"run: {run.config.name}   models: {', '.join(sorted(run.models))}")
    for side, agents in (("sellers", run.sellers), ("buyers", run.buyers)):
        cast = getattr(run.config.casting, side)
        kinds: dict[str, int] = defaultdict(int)
        for a in agents:
            kinds[a.model or f"scripted:{a.strategy}"] += 1
        seats = "all" if cast.mode == "all" else cast.per_market
        print(f"  {side:7}: {len(agents)} agents, mode={cast.mode}, per_market={seats} "
              f"({', '.join(f'{k} x{v}' for k, v in sorted(kinds.items()))})")
    for a in run.sellers + run.buyers:
        a.persona_body()          # raises if a persona file is missing


def print_plan(run: LoadedRun) -> None:
    plans = plan_markets(run.config.casting, run.sellers, run.buyers, run.config.seed,
                         pads=run.pads)
    print(f"{len(plans)} markets")
    for p in plans:
        print(f"  {p.market_id}: {len(p.sellers)} sellers {[a.id for a in p.sellers]}")
        print(f"  {'':{len(p.market_id)}}  {len(p.buyers)} buyers  {[a.id for a in p.buyers]}")


def print_summary(result: dict, top: int = 25) -> None:
    """The standing display: students first (all their agents pooled), then TA agents.
    Raw = seller profit / buyer surplus in dollars per market; grade from the TA anchors."""
    out = result["out"]
    print(f"wrote {out}  ({result['markets']} markets)")
    stand = result.get("standings") or {}
    f = lambda x: "-" if x is None else f"{x:.3f}"
    m = lambda x: "-" if x is None else f"{x:.2f}"
    for role, per_e in (stand.get("anchors") or {}).items():
        print(f"\ngrade anchors ({role}s): " + "; ".join(
            f"economy {e}: {a['low']['agent']} {a['low']['raw']:.2f} -> {a['low']['score']:g}, "
            f"{a['high']['agent']} {a['high']['raw']:.2f} -> {a['high']['score']:g}" for e, a in per_e.items()))
    for role in ("seller", "buyer"):
        rows = [s for s in stand.get("students", []) if s["role"] == role]
        if rows:
            graded = any("grade" in s for s in rows)
            print(f"\nstudent {role}s ({len(rows)}; raw = {'profit' if role == 'seller' else 'surplus'} "
                  f"$ per market{', grade from TA anchors, not capped' if graded else ''})")
            print(f"{'rank':>4}  {'student':24} {'agents':>6} {'mkts':>5} " + (f"{'grade':>7} " if graded else "")
                  + f"{'raw $':>9} {'se':>7} {'$/round':>8}")
            for s in rows[:top]:
                print(f"{s['rank']:>4}  {s['owner']:24} {s['agents']:>6} {s['markets']:>5} "
                      + (f"{f(s.get('grade')):>7} " if graded else "")
                      + f"{m(s['mean_raw']):>9} {m(s['se_raw']):>7} {m(s.get('mean_raw_per_round')):>8}")
            if len(rows) > top:
                print(f"      ... {len(rows) - top} more")
        ta = [a for a in stand.get("agents", []) if a["role"] == role and a["owner"] == "ta"]
        if ta:
            print(f"\nTA {role}s")
            for a in ta[:top]:
                print(f"      {a['agent']:24} {'':>6} {a['markets']:>5} {m(a['mean_raw']):>9} {m(a['se_raw']):>7}")
    if result.get("stats"):
        from adlmarket.report import stats_lines
        print()
        for line in stats_lines(result["stats"]):
            print(line)
    print(f"\nfull standings: {out}/standings.md   market health: {out}/economy_stats.json   "
          f"per market: {out}/markets/<id>/transcript.md")


def execute(args: argparse.Namespace, build: Callable[[], LoadedRun]) -> int:
    """Build the run, then validate (--dry-run) or play, per the flags."""
    try:
        if args.resume:
            from adlmarket.setups import resume_run
            run = resume_run(args.resume)
        else:
            run = build()
        describe(run)
        if args.dry_run:
            print_plan(run)
            return 0
        from adlmarket.engine import Runner
        out = Path(args.resume) if args.resume else None
        runner = Runner(run, mock_llm=args.mock_llm, out_dir=out,
                        invocation=" ".join(["python", *sys.argv]), resume=bool(args.resume))
        started = runner.out
        if args.mock_llm or not any(a.controller == "llm" for a in run.everyone()):
            print_summary(runner.run_all())
            return 0
        check_credentials(run)
        with ExitStack() as stack:
            if args.launch_vllm:
                if not run.serving:
                    raise ValueError("--launch-vllm needs a `serving` section in the models file")
                stack.enter_context(VLLMCluster(run, runner.out / "vllm"))
            n = register_adapters(run)       # renamed copies: <repo>/.cache/adapters, shared
            if n:
                print(f"registered {n} LoRA adapter(s)")
            print_summary(runner.run_all())
        return 0
    except KeyboardInterrupt:
        out = locals().get("started")
        print("\nstopped." + (f" Finished markets are saved; continue with:\n  python "
                              f"{Path(sys.argv[0]).name} --resume {out}"
                              + (" --mock-llm" if args.mock_llm else "")
                              if out and Path(out).exists() else ""), file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
