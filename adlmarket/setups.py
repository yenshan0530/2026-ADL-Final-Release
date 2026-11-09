"""The ways to set up a run, behind `run_economy.py` and `run_market.py`.

economy -- the whole class at once. Each economy partitions the graded agents into
           sub-markets; which agent sets play depends on the mode:
             --phase 1  student sellers (graded) + TA buyers
             --phase 2  TA sellers + student buyers (graded)
             --phase 3  student sellers + student buyers (both graded)
           The four sets {ta, student} x {sellers, buyers} are listed in the config's
           `agents:` section (see resolve_agent_set for the entry syntax).
market  -- one market with exactly the sellers and buyers listed, played `repeats` times.

All turn their YAML into the same RunConfig, so the engine, logs and scores are
identical either way.
"""

from __future__ import annotations

import copy
import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from adlmarket.config import (Casting, GradingSpec, LoadedRun, RunConfig, SideCasting, _read_yaml,
                              _Strict, load_agent, resolve_run)

Mode = Literal["seller", "buyer", "stage3"]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ------------------------------------------------------------------------- economy ---

def _rename_legacy(data: Any, names: dict[str, str]) -> Any:
    """Accept an older key (e.g. `economies`) for its current name (e.g. `repeats`)."""
    if isinstance(data, dict):
        for old, new in names.items():
            if old in data:
                if new in data:
                    raise ValueError(f"give `{new}` (the current name of `{old}`), not both")
                data = {**data, new: data[old]}
                del data[old]
    return data


class _RunFile(_Strict):
    """Fields shared by both run files."""

    name: str
    seed: int = 0
    models: list[str] = Field(default_factory=lambda: ["configs/models.yaml"])
    market: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "results"     # runs go to <output_dir>/<script>_<timestamp>/
    max_workers: int = 8
    parallel_markets: int = 1           # see RunConfig
    min_call_interval: float = 0.0
    call_timeout: float = 120.0         # see RunConfig
    call_retries: int = 4
    max_failed_calls_in_a_row: int = 8
    save_trajectories: bool = True      # every turn of every agent; see RunConfig
    save_prompts: bool = False          # see RunConfig

    @field_validator("models", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


class TAPools(_Strict):
    """Legacy `ta:` section (sellers / buyers lists); `agents: {ta: ...}` replaces it."""
    sellers: list[str] = Field(default_factory=list)
    buyers: list[str] = Field(default_factory=list)


class AgentSets(_Strict):
    """One side of `agents:`: seller and buyer entries (each empty unless given). An entry
    is a string (path / glob / re:regex) or {agent: <string>, personas: [...], lora: ...}."""
    sellers: list[str | dict[str, Any]] = Field(default_factory=list)
    buyers: list[str | dict[str, Any]] = Field(default_factory=list)

    @field_validator("sellers", "buyers", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        return [] if v is None else [v] if isinstance(v, str) else v


class AgentsSection(_Strict):
    ta: AgentSets = Field(default_factory=AgentSets)
    student: AgentSets = Field(default_factory=AgentSets)


class ModeOverrides(_Strict):
    """Per-phase section (`phase1:`, `phase2:`, `phase3:`): merged over the top level."""
    market: dict[str, Any] = Field(default_factory=dict)
    sellers_per_market: Optional[int] = None
    buyers_per_market: Optional[int] = None


class EconomyConfig(_RunFile):
    name: str = "economy"
    # Who plays: {ta, student} x {sellers, buyers}; every set is empty unless given.
    agents: Optional[AgentsSection] = None
    # Legacy form of `agents` (student folders pattern + TA pools); used when `agents` is absent.
    students: Optional[str] = None
    ta: Optional[TAPools] = None
    # Personas each student agent plays: its LoRA with its personas 1..N (persona_1.md ...);
    # a student with fewer is padded with persona 1, with none it plays an empty persona.
    personas_per_student: int = Field(default=3, ge=1)
    sellers_per_market: int = 5
    buyers_per_market: int = 10
    market_count: Literal["min", "max"] = "min"
    # Student sellers: exactly sellers_per_market per market; markets short of students are
    # topped up with TA sellers (agents: ta: sellers). False: the market_count rule.
    pad_sellers: bool = True
    # Student buyers: a market with fewer than buyers_per_market of them (a small class, or
    # a test with few agents) is topped up with TA buyers (agents: ta: buyers); not graded.
    pad_buyers: bool = True
    # Play each market once per seller seat with the stock bundles rotated (Casting.rotate).
    rotate: bool = True
    # Phases 1 and 2: the TA agents of the graded role (agents: ta:) play among
    # the students, in the same markets, and their scores set the grade anchors (grading).
    ta_anchors: bool = True
    grading: GradingSpec = Field(default_factory=GradingSpec)
    repeats: int = Field(default=3, ge=1)   # economies to play: each re-partitions the class
    save_trajectories: bool = False     # the whole class x economies is large; off by default
    phase1: ModeOverrides = Field(default_factory=ModeOverrides)   # --phase 1
    phase2: ModeOverrides = Field(default_factory=ModeOverrides)   # --phase 2
    phase3: ModeOverrides = Field(default_factory=ModeOverrides)   # --phase 3

    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, data: Any) -> Any:
        return _rename_legacy(data, {"economies": "repeats", "test_seller": "phase1",
                                     "test_buyer": "phase2", "stage3": "phase3"})

    def agent_sets(self) -> AgentsSection:
        if self.agents is not None:
            if self.students is not None or self.ta is not None:
                raise ValueError("give either `agents:` or the legacy `students:` / `ta:`, not both")
            return self.agents
        students = [self.students] if self.students else []
        ta = self.ta or TAPools()
        return AgentsSection(ta=AgentSets(sellers=ta.sellers, buyers=ta.buyers),
                             student=AgentSets(sellers=students, buyers=students))


# ------------------------------------------------------------- agent set entries ---

_GLOB_CHARS = re.compile(r"[*?\[]")
_REGEX_CHARS = set("\\^$.|?*+()[]{}")
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


@dataclass
class AgentSet:
    """The agent files one set resolved to, and what was left out (reported, never silent)."""
    files: list[str] = field(default_factory=list)       # repo-relative POSIX paths
    entries: list = field(default_factory=list)          # for the casting: path or mapping
    no_agent: list[str] = field(default_factory=list)    # matched folders without a <role> agent
    other_role: list[str] = field(default_factory=list)  # matched files of the other role


def _relative(entry: str) -> str:
    rel = entry.strip()
    if not rel:
        raise ValueError("empty agent entry")
    p = Path(rel.removeprefix("re:"))
    if p.is_absolute() or rel.startswith("~"):
        raise ValueError(f"agent entry {entry!r}: use a path relative to the repo root, "
                         f"not an absolute one")
    if ".." in p.parts:
        raise ValueError(f"agent entry {entry!r}: '..' is not allowed; paths are relative "
                         f"to the repo root")
    return rel


def _regex_matches(pattern: str, root: Path) -> list[str]:
    r"""Repo-relative paths (files and folders) that fully match `pattern`. The walk starts
    at the pattern's literal leading folders, so `re:agents/s\d+` only walks agents/."""
    rx = re.compile(pattern)
    prefix = []
    for part in pattern.split("/")[:-1]:
        if any(c in _REGEX_CHARS for c in part):
            break
        prefix.append(part)
    start = root.joinpath(*prefix)
    if not start.is_dir():
        return []
    out = []
    for d, dirs, files in os.walk(start):
        dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS and not x.startswith("."))
        for name in dirs + sorted(files):
            rel = (Path(d) / name).relative_to(root).as_posix()
            if rx.fullmatch(rel):
                out.append(rel)
    return sorted(out)


def _agent_file_in(folder: Path, role: str) -> Optional[Path]:
    """A folder entry: a student folder (its <role>/ folder, or agents/<role>/ in a
    submission: agent.yaml, checkpoint/ or a persona file; agent.yaml may be missing), a role
    folder itself, or an agent folder (<folder>/agent.yaml)."""
    from adlmarket.config import is_agent_folder
    for f in (folder / role, folder / "agents" / role, folder):    # repo / submission / role
        if is_agent_folder(f) and f.name == role:
            return f / "agent.yaml"
    if (folder / "agent.yaml").is_file():
        return folder / "agent.yaml"
    return None


def resolve_agent_set(entries: list, role: str, root: Path) -> AgentSet:
    r"""Turn a set's entries into agent files. Every entry is relative to the repo root, and
    may be (mixed freely in one list):

      agents/alice/seller/agent.yaml        a path: an agent file, or a folder holding
      agents/alice                          <role>/agent.yaml (student) or agent.yaml
      agents/[!_]*                          a glob (`**` recurses)
      re:agents/s\d+/seller/agent\.yaml     a regex (after `re:`), matched against the whole
                                            repo-relative path of files and folders

    or a mapping that also sets the personas and LoRA of every agent it matches (both
    relative to each agent file):

      {agent: "agents/[!_]*", personas: ["persona_*.md"], lora: checkpoint}

    A path must exist. Globs and regexes may also match folders: a matched folder with no
    agent for `role` is listed in `no_agent`, a matched file of the other role in
    `other_role` (both reported by the caller). The same file twice counts once.
    """
    out = AgentSet()
    seen: set[str] = set()

    over: dict[str, Any] = {}

    def add(path: Path, explicit: bool, entry: str) -> None:
        rel = path.relative_to(root).as_posix()
        if path.name == "agent.yaml" and not path.exists():
            path = path.parent                   # a role folder whose agent.yaml is optional
        if path.is_dir():
            f = _agent_file_in(path, role)
            if f is None:
                if explicit:
                    raise ValueError(f"agent entry {entry!r}: folder has no {role}/agent.yaml "
                                     f"or agent.yaml")
                out.no_agent.append(rel)
                return
            path, rel = f, f.relative_to(root).as_posix()
        if rel in seen:
            return
        if path.suffix not in (".yaml", ".yml"):
            if explicit:
                raise ValueError(f"agent entry {entry!r}: not a YAML file")
            return
        agent_role = load_agent(path).role      # (a missing role-folder agent.yaml is fine)
        if agent_role != role:
            if explicit:
                raise ValueError(f"agent entry {entry!r}: has role {agent_role}, listed as {role}")
            out.other_role.append(rel)
            return
        seen.add(rel)
        out.files.append(rel)
        out.entries.append({"agent": glob.escape(rel), **over} if over else glob.escape(rel))

    for entry in entries:
        over = {}
        if isinstance(entry, dict):
            unknown = set(entry) - {"agent", "personas", "lora"}
            if unknown or "agent" not in entry:
                raise ValueError(f"agent entry {entry}: needs `agent`, may set `personas` and "
                                 f"`lora` (unknown: {sorted(unknown)})")
            over = {k: entry[k] for k in ("personas", "lora") if k in entry}
            entry = entry["agent"]
        rel = _relative(entry)
        if rel.startswith("re:"):
            matches = _regex_matches(rel[3:], root)
            if not matches:
                raise ValueError(f"agent entry {entry!r}: the regex matches nothing")
            for m in matches:
                add(root / m, False, entry)
        elif _GLOB_CHARS.search(rel):
            matches = sorted(glob.glob(str(root / rel), recursive=True))
            if not matches:
                raise ValueError(f"agent entry {entry!r}: the glob matches nothing")
            for m in matches:
                add(Path(m), False, entry)
        else:
            path = root / rel
            if not path.exists():
                raise ValueError(f"agent entry {entry!r}: no such file or folder")
            add(path, True, entry)
    return out


# ------------------------------------------------------------------------- economy ---

# mode -> (seller set, buyer set); the student side(s) are partitioned (graded), TA sampled
MODE_SETS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "seller": (("student", "sellers"), ("ta", "buyers")),
    "buyer": (("ta", "sellers"), ("student", "buyers")),
    "stage3": (("student", "sellers"), ("student", "buyers")),
}
MODE_SECTION = {"seller": "phase1", "buyer": "phase2", "stage3": "phase3"}
PHASE_MODE = {1: "seller", 2: "buyer", 3: "stage3"}


@dataclass
class EconomyInfo:
    """What an economy run loaded, for the `N TA buyers loaded, ...` report."""
    mode: str
    sets: dict[str, AgentSet]           # "student sellers" -> its AgentSet (the two in play)
    markets_per_economy: int
    economies: int

    agents: dict[str, list] = field(default_factory=dict)   # set name -> loaded AgentSpecs
    expected_personas: int = 0
    rotations: int = 1
    padding: Optional[tuple[int, int]] = None   # (seats filled by TA sellers per economy, pool)
    buyer_padding: Optional[tuple[int, int]] = None   # the same for TA buyers

    def lines(self) -> list[str]:
        out = []
        for name, st in self.sets.items():
            agents = self.agents.get(name, [])
            per: dict[str, int] = {}
            for a in agents:
                per[a.variant_of or a.id] = per.get(a.variant_of or a.id, 0) + 1
            counts = sorted(set(per.values()))
            how = (f" ({len(st.files)} files x {counts[0]} personas)" if len(counts) == 1 and counts[0] > 1
                   else f" ({len(st.files)} files, {'/'.join(map(str, counts))} personas)" if len(counts) > 1
                   else "")
            out.append(f"loaded {len(agents)} {name}{how}")
            if name.startswith("student") and self.expected_personas:
                given = {a.variant_of or a.id: a.personas_given for a in agents}
                short = sorted(f"{b} ({g or 0})" for b, g in given.items()
                               if (g or 0) < self.expected_personas)
                if short:
                    out.append(f"  note: {len(short)} agent(s) handed in fewer than "
                               f"{self.expected_personas} personas (padded with persona 1; none: "
                               f"empty persona): {', '.join(short)}")
            if st.no_agent:
                out.append(f"  warning: {len(st.no_agent)} matched folder(s) have no agent for "
                           f"this role: {', '.join(st.no_agent)}")
            if st.other_role:
                out.append(f"  note: {len(st.other_role)} matched file(s) of the other role "
                           f"left out: {', '.join(st.other_role)}")
        if self.mode == "stage3":
            ns = len(self.agents.get("student sellers", []))
            nb = len(self.agents.get("student buyers", []))
            if ns and nb and ns % nb and nb % ns:
                n = self.markets_per_economy
                sizes = sorted({nb // n, -(-nb // n)})
                out.append(f"  warning: {ns} student sellers and {nb} student buyers: neither is "
                           f"a whole multiple of the other, so markets are uneven (buyers per "
                           f"market: {' or '.join(map(str, sizes))} over {n} markets); every "
                           f"agent still plays one market per economy, but students face "
                           f"different numbers of counterparts")
        if self.padding:
            seats, pool = self.padding
            out.append(f"padding {seats} seller seat(s) per economy with TA sellers (pool of {pool})")
        if self.buyer_padding:
            seats, pool = self.buyer_padding
            out.append(f"padding {seats} buyer seat(s) per economy with TA buyers (pool of {pool}; "
                       f"markets short of {'student ' if self.mode != 'seller' else ''}buyers)")
        total = self.markets_per_economy * self.rotations * self.economies
        rot = f" x {self.rotations} rotations" if self.rotations > 1 else ""
        out.append(f"creating {self.markets_per_economy} partition(s) (markets) per economy{rot} "
                   f"x {self.economies} economies = {total} markets")
        return out

    @property
    def missing(self) -> list[str]:
        """Folder names (student ids) matched without an agent for their graded role."""
        return [Path(m).name for name, st in self.sets.items()
                if name.startswith("student") for m in st.no_agent]


def _agents_by_set(loaded: dict[str, "AgentSet"], run: LoadedRun) -> dict[str, list]:
    """The loaded agents of each named set (persona copies included)."""
    out = {}
    for name, st in loaded.items():
        side = run.sellers if "sellers" in name else run.buyers
        files = {str((run.root / f).resolve()) for f in st.files}
        out[name] = [a for a in side if a.source and str(Path(a.source).resolve()) in files]
    return out


def _buyer_padding(casting: Casting, run: LoadedRun) -> Optional[tuple[int, int]]:
    """TA buyers topping up markets short of buyers_per_market student buyers."""
    from adlmarket.casting import markets_per_repeat
    side = casting.buyers
    if side.mode != "partition" or not run.pads.get("buyers"):
        return None
    n = markets_per_repeat(casting, run.sellers, run.buyers)
    q, extra = divmod(len(run.buyers), n)
    seats = sum(max(0, side.per_market - (q + (i < extra))) for i in range(n))
    return (seats, len(run.pads["buyers"])) if seats else None


def _padding(casting: Casting, run: LoadedRun) -> Optional[tuple[int, int]]:
    from adlmarket.casting import markets_per_repeat
    side = casting.sellers
    if side.mode != "partition" or not side.exact:
        return None
    n = markets_per_repeat(casting, run.sellers, run.buyers)
    seats = n * side.per_market - len(run.sellers)
    return (seats, len(run.pads.get("sellers", []))) if seats else None


def economy_run(path: str | Path, mode: Mode, *, root: Optional[Path] = None,
                sellers_per_market: Optional[int] = None,
                buyers_per_market: Optional[int] = None,
                repeats: Optional[int] = None, seed: Optional[int] = None,
                save_trajectories: Optional[bool] = None,
                info: Optional[list] = None,
                personas_per_student: Optional[int] = None,
                rotate: Optional[bool] = None,
                ta_anchors: Optional[bool] = None,
                ) -> tuple[LoadedRun, list[str]]:
    """Build an economy run for `mode` (seller | buyer | stage3) from the config file. The
    keyword settings override the file from Python (tests); the command line has none.
    Returns the run and the students skipped for having no agent of the graded role. Pass
    `info=[]` to also get the EconomyInfo appended to it."""
    root = Path(root) if root else Path.cwd()
    cfg = EconomyConfig(**_read_yaml(Path(path)))
    given = {k: v for k, v in {
        "repeats": repeats, "seed": seed, "save_trajectories": save_trajectories,
        "personas_per_student": personas_per_student, "rotate": rotate,
        "ta_anchors": ta_anchors}.items() if v is not None}
    cfg = cfg.model_copy(update=given)
    # Market sizes: keyword > the phase's section > top level. A keyword size is written to
    # the top level and cleared from the phase section, so the saved config keeps it.
    section = MODE_SECTION[mode]
    cli = {k: v for k, v in (("sellers_per_market", sellers_per_market),
                             ("buyers_per_market", buyers_per_market)) if v is not None}
    if cli:
        cfg = cfg.model_copy(update={**cli, section: getattr(cfg, section).model_copy(
            update={k: None for k in cli})})
    overridden = bool(given or cli)
    overrides: ModeOverrides = getattr(cfg, section)
    s_per = overrides.sellers_per_market or cfg.sellers_per_market
    b_per = overrides.buyers_per_market or cfg.buyers_per_market
    sets = cfg.agent_sets()

    casting_sides, loaded = [], {}
    pad_info: Optional[AgentSet] = None
    for (who, side), per in zip(MODE_SETS[mode], (s_per, b_per)):
        role = side[:-1]
        entries = getattr(getattr(sets, who), side)
        name = f"{'TA' if who == 'ta' else 'student'} {side}"
        if not entries:
            raise ValueError(f"--phase {MODE_SECTION[mode][-1]} needs {name}: list them "
                             f"under agents: {who}: {side}: in {path}")
        st = resolve_agent_set(entries, role, root)
        if not st.files:
            raise ValueError(f"no {name} found (entries: {entries})")
        loaded[name] = st
        if who == "student" and mode != "stage3" and cfg.ta_anchors:
            ta_entries = getattr(sets.ta, side)
            if not ta_entries:
                raise ValueError(f"ta_anchors: the TA {side} play among the student {side} to set the "
                                 f"grade anchors; list them under agents: ta: {side}: in {path} "
                                 f"(or set ta_anchors: false)")
            anchors = resolve_agent_set(ta_entries, role, root)
            loaded[f"TA {side} (grade anchors, playing among the students)"] = anchors
            st = AgentSet(files=st.files + anchors.files, entries=st.entries + anchors.entries,
                          no_agent=st.no_agent, other_role=st.other_role)
        exact, pad = False, []
        if who == "student" and side == "sellers" and cfg.pad_sellers:
            exact = True
            if sets.ta.sellers:
                pad_info = resolve_agent_set(sets.ta.sellers, "seller", root)
                pad = pad_info.entries
        if who == "student" and side == "buyers" and cfg.pad_buyers and sets.ta.buyers:
            pad = resolve_agent_set(sets.ta.buyers, "buyer", root).entries
        casting_sides.append(SideCasting(agents=st.entries, per_market=per,
                                         mode="partition" if who == "student" else "sample",
                                         exact=exact, pad=pad))
    names = list(loaded)
    seller_files = {f for n in names if "sellers" in n for f in loaded[n].files}
    buyer_files = {f for n in names if "buyers" in n for f in loaded[n].files}
    overlap = seller_files & buyer_files
    if overlap:
        raise ValueError(f"agent file(s) on both sides: {sorted(overlap)}")

    casting = Casting(sellers=casting_sides[0], buyers=casting_sides[1], repeats=cfg.repeats,
                      market_count=cfg.market_count, rotate=cfg.rotate)
    run_cfg = RunConfig(
        name=f"{cfg.name}_{'stage3' if mode == 'stage3' else 'test_' + mode}", seed=cfg.seed,
        models=cfg.models, market=deep_merge(cfg.market, overrides.market),
        casting=casting,
        output_dir=cfg.output_dir, max_workers=cfg.max_workers, parallel_markets=cfg.parallel_markets,
        save_prompts=cfg.save_prompts,
        min_call_interval=cfg.min_call_interval, call_timeout=cfg.call_timeout,
        call_retries=cfg.call_retries, max_failed_calls_in_a_row=cfg.max_failed_calls_in_a_row,
        save_trajectories=cfg.save_trajectories,
        personas_per_student=cfg.personas_per_student,
        graded_roles={"seller": ["seller"], "buyer": ["buyer"], "stage3": ["seller", "buyer"]}[mode],
        grading=cfg.grading if (mode != "stage3" and cfg.ta_anchors) else None, source=str(path),
        origin={"script": "run_economy.py", "phase": int(MODE_SECTION[mode][-1]),
                "overridden": overridden,
                "file": cfg.model_dump(mode="json", exclude_defaults=True)})
    run = resolve_run(run_cfg, root)
    run_cfg.check_items()
    from adlmarket.casting import markets_per_repeat
    econ = EconomyInfo(mode=mode, sets=loaded, economies=cfg.repeats,
                       markets_per_economy=markets_per_repeat(casting, run.sellers, run.buyers),
                       agents=_agents_by_set(loaded, run),
                       expected_personas=cfg.personas_per_student,
                       rotations=s_per if cfg.rotate and s_per > 1 else 1,
                       padding=_padding(casting, run), buyer_padding=_buyer_padding(casting, run))
    if econ.padding and not econ.padding[1]:
        raise ValueError(f"{len(run.sellers)} student sellers do not fill markets of "
                         f"{s_per}: {econ.padding[0]} seat(s) need TA sellers to pad them; list "
                         f"them under agents: ta: sellers: in {path} (or set pad_sellers: false)")
    if info is not None:
        info.append(econ)
    return run, econ.missing


# -------------------------------------------------------------------------- market ---

class MarketConfig(_RunFile):
    name: str = "market"
    # Personas each student agent plays (as in economy configs); 1 = persona 1 only, so the
    # market has exactly the agents listed. TA agents are never copied.
    personas_per_student: int = Field(default=1, ge=1)
    # Play each repeat once per seller seat with the stock rotated one seat each time,
    # so every seller holds every seller's stock once (as in economies; Casting.rotate).
    rotate: bool = True
    # agent files / role folders / globs, or {agent, personas, lora} mappings (as in
    # economy.yaml); everyone listed is in the market
    sellers: list[str | dict[str, Any]]
    buyers: list[str | dict[str, Any]]
    repeats: int = Field(default=1, ge=1)   # independent plays (fresh goods + seed each time)

    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, data: Any) -> Any:
        return _rename_legacy(data, {"simulations": "repeats"})


def market_run(path: str | Path, *, root: Optional[Path] = None,
               repeats: Optional[int] = None, seed: Optional[int] = None,
               save_trajectories: Optional[bool] = None,
               personas_per_student: Optional[int] = None,
               rotate: Optional[bool] = None) -> LoadedRun:
    """Build a run_market.py run from the config file; the keyword settings override the
    file from Python (tests), the command line has none."""
    root = Path(root) if root else Path.cwd()
    cfg = MarketConfig(**_read_yaml(Path(path)))
    given = {k: v for k, v in {
        "repeats": repeats, "seed": seed, "save_trajectories": save_trajectories,
        "personas_per_student": personas_per_student, "rotate": rotate}.items() if v is not None}
    cfg = cfg.model_copy(update=given)
    casting = Casting(sellers=SideCasting(agents=cfg.sellers, mode="all", per_market=len(cfg.sellers)),
                      buyers=SideCasting(agents=cfg.buyers, mode="all", per_market=len(cfg.buyers)),
                      repeats=cfg.repeats, rotate=cfg.rotate)
    run_cfg = RunConfig(name=cfg.name, seed=cfg.seed, models=cfg.models,
                        market=cfg.market, casting=casting, output_dir=cfg.output_dir,
                        max_workers=cfg.max_workers, parallel_markets=cfg.parallel_markets,
                        save_prompts=cfg.save_prompts, min_call_interval=cfg.min_call_interval,
                        call_timeout=cfg.call_timeout, call_retries=cfg.call_retries,
                        max_failed_calls_in_a_row=cfg.max_failed_calls_in_a_row,
                        save_trajectories=cfg.save_trajectories,
                        personas_per_student=cfg.personas_per_student,
                        source=str(path),
                        origin={"script": "run_market.py", "overridden": bool(given),
                                "file": cfg.model_dump(mode="json", exclude_defaults=True)})
    run = resolve_run(run_cfg, root)
    run_cfg.check_items()
    return run


# -------------------------------------------------------------------------- resume ---

def resume_run(run_dir: str | Path, *, root: Optional[Path] = None) -> LoadedRun:
    """The run of a stopped run folder, rebuilt from its config.yaml (the config it played)
    and the script and phase in its metadata.json. Nothing in a run is random except
    through `seed`, so the same markets come out; the runner checks them against plan.json."""
    import json
    root = Path(root) if root else Path.cwd()
    run_dir = Path(run_dir)
    config, meta = run_dir / "config.yaml", run_dir / "metadata.json"
    if not (config.exists() and meta.exists()):
        raise FileNotFoundError(f"{run_dir} is not a run folder (no config.yaml / metadata.json)")
    m = json.loads(meta.read_text(encoding="utf-8"))
    if m.get("script") == "run_economy.py":
        run, _ = economy_run(config, PHASE_MODE[int(m["phase"])], root=root)
    else:
        run = market_run(config, root=root)
    run.config.source = str(config)
    return run
