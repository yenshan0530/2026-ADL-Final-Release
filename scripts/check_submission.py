"""Check a final-project submission before handing it in (students) or when collecting it (TAs).

    python scripts/check_submission.py <student_id>.zip
    python scripts/check_submission.py <student_id>/          # the folder you will zip

Layout (docs/SUBMISSION.md):

    <student_id>.zip
    └── <student_id>/
        ├── training/              all your training code, with README.md (how to reproduce
        │                          your agents: the repository's template, filled in)
        └── agents/
            ├── seller/            checkpoint/ (adapter_config.json + adapter_model.safetensors),
            │                      persona_1.md, persona_2.md, ...; agent.yaml optional
            └── buyer/             the same

Each LoRA: PEFT, on Qwen/Qwen3.5-4B, rank <= 32, language-model layers only. No
__pycache__, *.pyc, __MACOSX, .DS_Store or base-model weights; report.pdf goes to NTU COOL.
Errors (the submission would be rejected) exit with status 1; warnings do not. At the end
the agents are loaded the way grading loads them and listed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                # the adlmarket package at the root

BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_RANK = 32
PERSONA_CHARS = 8000
PLAYED = 3                                   # personas_per_student in the grading config
ALLOWED_AGENT_KEYS = {"reasoning"}           # graded agents: everything else is fixed
EXPECTED = {"agent.yaml", "checkpoint"}
# Never in a submission: caches, OS litter, and the base model (graders have it)
JUNK_DIRS = {"__pycache__", "__MACOSX", ".ipynb_checkpoints"}
JUNK_FILES = {".DS_Store", "Thumbs.db"}
BASE_MODEL_FILE = re.compile(r"^(model|pytorch_model|consolidated)[-.0-9a-z_]*\.(safetensors|bin|pt|pth)$"
                             r"|\.safetensors\.index\.json$|\.gguf$")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _safetensors_header(path: Path) -> dict:
    raw = path.open("rb").read(8)
    n = int.from_bytes(raw, "little")
    with path.open("rb") as f:
        f.seek(8)
        header = json.loads(f.read(n))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def _safetensors_keys(path: Path) -> list[str]:
    return list(_safetensors_header(path))


def lora_tensor_rank(header: dict) -> int:
    """The largest rank among the LoRA tensors: lora_A is [r, in], lora_B is [out, r]."""
    ranks = [v["shape"][0] for k, v in header.items() if "lora_A" in k and v.get("shape")]
    ranks += [v["shape"][-1] for k, v in header.items() if "lora_B" in k and v.get("shape")]
    return max(ranks, default=0)


def check_lora(ck: Path, role: str, rep: Report) -> None:
    where = f"{role}/checkpoint"
    if not ck.is_dir():
        rep.error(f"{where}/ is missing (your LoRA: adapter_config.json + adapter_model.safetensors)")
        return
    cfg_file, weights = ck / "adapter_config.json", ck / "adapter_model.safetensors"
    if not cfg_file.is_file():
        rep.error(f"{where}/adapter_config.json is missing (written by PEFT save_pretrained)")
        return
    if not weights.is_file():
        if (ck / "adapter_model.bin").is_file():
            rep.error(f"{where}: adapter_model.bin found; hand in adapter_model.safetensors")
        else:
            rep.error(f"{where}/adapter_model.safetensors is missing")
        return
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        rep.error(f"{where}/adapter_config.json is not valid JSON: {exc}")
        return
    if cfg.get("peft_type", "LORA") != "LORA":
        rep.error(f"{where}: peft_type is {cfg.get('peft_type')!r}; only LoRA adapters are accepted")
    r = int(cfg.get("r") or 0)
    if not 0 < r <= MAX_RANK:
        rep.error(f"{where}: rank r={cfg.get('r')}; must be 1..{MAX_RANK}")
    high = {k: v for k, v in (cfg.get("rank_pattern") or {}).items() if int(v) > MAX_RANK}
    if high:
        rep.error(f"{where}: rank_pattern gives some layers rank above {MAX_RANK}: {high}")
    base = cfg.get("base_model_name_or_path") or ""
    if Path(base).name.lower() != Path(BASE_MODEL).name.lower():
        rep.error(f"{where}: trained on {base!r}; must be {BASE_MODEL}")
    if cfg.get("use_dora"):
        rep.error(f"{where}: DoRA (use_dora: true) cannot be served by vLLM; train a plain LoRA")
    if cfg.get("modules_to_save"):
        rep.error(f"{where}: modules_to_save={cfg['modules_to_save']}: only LoRA layers are allowed, "
                  f"no full copies of modules")
    if cfg.get("use_rslora"):
        rep.warn(f"{where}: use_rslora is set; check that your adapter behaves the same on vLLM")
    try:
        header = _safetensors_header(weights)
    except (OSError, ValueError) as exc:
        rep.error(f"{where}/adapter_model.safetensors cannot be read: {exc}")
        return
    keys = list(header)
    tensor_rank = lora_tensor_rank(header)
    if tensor_rank > MAX_RANK:
        rep.error(f"{where}: the LoRA weights have rank {tensor_rank} (tensor shapes); must be "
                  f"at most {MAX_RANK}")
    elif tensor_rank and r and tensor_rank != r and not cfg.get("rank_pattern"):
        rep.warn(f"{where}: adapter_config.json says r={r} but the weights have rank "
                 f"{tensor_rank}; save the adapter with PEFT save_pretrained")
    if not keys:
        rep.error(f"{where}/adapter_model.safetensors holds no tensors")
    bad = [k for k in keys if ".visual." in k or "embed_tokens" in k or "lm_head" in k]
    if bad:
        rep.error(f"{where}: adapts the vision encoder / embeddings / lm_head ({bad[0]} ...); "
                  f"only the language model's layers are allowed")
    if not any("lora_A" in k for k in keys) or not any("lora_B" in k for k in keys):
        rep.error(f"{where}: no lora_A / lora_B tensors: not a LoRA adapter")
    mb = weights.stat().st_size / 1e6
    if mb > 800:                      # a rank-32 LoRA on every layer is about 250 MB
        rep.warn(f"{where}: adapter_model.safetensors is {mb:.0f} MB, unusually large for rank <= 32")


def check_personas(folder: Path, role: str, rep: Report) -> list[Path]:
    files = sorted(folder.glob("persona_*.md"), key=lambda p: _natural(p.name))
    if not files and (folder / "persona.md").is_file():
        files = [folder / "persona.md"]
    others = [p for p in folder.glob("*.md") if p not in files]
    for p in others:
        rep.warn(f"{role}/{p.name} is not a persona file (name them persona_1.md, persona_2.md, ...)")
    if not files:
        rep.warn(f"{role}: no persona files; your LoRA will play with an empty persona")
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            rep.error(f"{role}/{p.name} is not UTF-8 text")
            continue
        if len(text) > PERSONA_CHARS:
            rep.error(f"{role}/{p.name} is {len(text)} characters; the limit is {PERSONA_CHARS}")
        if not text.strip():
            rep.warn(f"{role}/{p.name} is empty")
    if 0 < len(files) < PLAYED:
        rep.warn(f"{role}: {len(files)} persona(s); grading plays {PLAYED}, so persona 1 fills the rest")
    if len(files) > PLAYED:
        rep.warn(f"{role}: {len(files)} personas; grading plays the first {PLAYED} "
                 f"({', '.join(p.name for p in files[:PLAYED])})")
    return files


def check_agent_yaml(folder: Path, role: str, rep: Report) -> None:
    f = folder / "agent.yaml"
    if not f.is_file():
        return
    import yaml
    try:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        rep.error(f"{role}/agent.yaml is not valid YAML: {exc}")
        return
    if not isinstance(data, dict):
        rep.error(f"{role}/agent.yaml must be a mapping (e.g. `reasoning: false`)")
        return
    extra = sorted(set(data) - ALLOWED_AGENT_KEYS)
    if extra:
        rep.error(f"{role}/agent.yaml sets {', '.join(extra)}; graded agents may only set "
                  f"{', '.join(sorted(ALLOWED_AGENT_KEYS))} (the model, endpoint, LoRA and "
                  f"personas come from the folder)")
    if "reasoning" in data and not isinstance(data["reasoning"], bool):
        rep.error(f"{role}/agent.yaml: reasoning must be true or false")


def check_role(folder: Path, role: str, rep: Report) -> None:
    if not folder.is_dir():
        rep.error(f"{role}/ is missing")
        return
    check_lora(folder / "checkpoint", role, rep)
    personas = check_personas(folder, role, rep)
    check_agent_yaml(folder, role, rep)
    known = EXPECTED | {p.name for p in personas} | {p.name for p in folder.glob("*.md")}
    for p in sorted(folder.iterdir()):
        if p.name not in known:
            rep.warn(f"{role}/{p.name}: not part of a submission (ignored; leave it out)")
    ck = folder / "checkpoint"
    if ck.is_dir():
        for p in sorted(ck.iterdir()):
            if p.name not in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
                rep.warn(f"{role}/checkpoint/{p.name}: not needed (tokenizer and training files are "
                         f"ignored; the base model's are used)")


def check_readme(readme: Path, rep: Report) -> None:
    """training/README.md: the template from the repository, filled in."""
    if not readme.is_file():
        rep.error("training/README.md is missing (how to reproduce your agents: start from "
                  "the template in the repository's training/README.md)")
        return
    text = readme.read_text(encoding="utf-8", errors="replace")
    for placeholder, what in (("<Title of your approach>", "the title"),
                              ("<student ID> <your name>", "the author (student ID and name)")):
        if placeholder in text:
            rep.error(f"training/README.md: fill in {what}")
    unanswered = text.count("*your answer*")
    if unanswered:
        rep.warn(f"training/README.md: {unanswered} question(s) still say *your answer*")


def check_layout(sub: Path, rep: Report) -> None:
    """training/         (with README.md), agents/{seller,buyer}; nothing that must not be there."""
    ts = sub / "training"
    if not ts.is_dir():
        rep.error("training/         is missing (all the code you used to train your LoRAs, "
                  "and README.md)")
    else:
        if not any(p.is_file() and p.name != "README.md" for p in ts.rglob("*")):
            rep.error("training/         has no training code")
        check_readme(ts / "README.md", rep)
    if not (sub / "agents").is_dir():
        if (sub / "seller").is_dir() or (sub / "buyer").is_dir():
            rep.error("seller/ and buyer/ go under agents/: <student_id>/agents/seller/, "
                      "<student_id>/agents/buyer/")
        else:
            rep.error("agents/ is missing (agents/seller/ and agents/buyer/)")
    for p in sorted(sub.rglob("*")):
        rel = p.relative_to(sub).as_posix()
        if p.is_dir() and p.name in JUNK_DIRS:
            rep.error(f"{rel}/: remove it (caches and OS files must not be submitted)")
        elif p.is_file():
            if p.name in JUNK_FILES or p.suffix == ".pyc":
                rep.error(f"{rel}: remove it (caches and OS files must not be submitted)")
            elif BASE_MODEL_FILE.search(p.name) and p.name != "adapter_model.safetensors":
                rep.error(f"{rel}: looks like base-model weights; submit only your LoRA adapters")
            elif p.name.lower() == "report.pdf":
                rep.warn(f"{rel}: hand in your report on NTU COOL, not in the zip")
            elif p.stat().st_size > 1_500_000_000:
                rep.error(f"{rel} is {p.stat().st_size / 1e9:.1f} GB: too large for a LoRA submission")


def preview(sub: Path, rep: Report) -> None:
    """Load the folder the way grading does and list the agents that would play."""
    from adlmarket.config import load_agent, persona_variants
    for role in ("seller", "buyer"):
        folder = sub / "agents" / role
        if not folder.is_dir():
            continue
        try:
            agent = load_agent(folder / "agent.yaml")
            copies = persona_variants(agent, PLAYED)
        except Exception as exc:                 # noqa: BLE001 - report, do not crash
            rep.error(f"{role}: the harness cannot load it: {exc}")
            continue
        print(f"  {role}: LoRA {agent.lora or '(none: plain base model)'}, thinking "
              f"{'on' if agent.reasoning else 'off'}")
        for c in copies:
            print(f"    {c.id}: persona {c.persona or '(empty)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="the submission folder, or its .zip")
    a = ap.parse_args()
    path = Path(a.path)
    rep = Report()
    tmp = None
    try:
        if path.suffix == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="submission_"))
            with zipfile.ZipFile(path) as z:
                if any(n.startswith("__MACOSX") or "/.DS_Store" in n or n == ".DS_Store" for n in z.namelist()):
                    rep.error("the zip holds __MACOSX/ or .DS_Store files: zip without them "
                              "(e.g. `zip -r <id>.zip <id> -x '*.DS_Store' -x '__MACOSX/*'`)")
                names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
                tops = {n.split("/")[0] for n in names if n.strip("/")}
                if len(tops) != 1:
                    print(f"error: the zip must hold one folder named after your student ID, found "
                          f"{sorted(tops)}")
                    return 1
                for n in names:
                    target = (tmp / n).resolve()
                    if not target.is_relative_to(tmp.resolve()):
                        print(f"error: unsafe path in the zip: {n}")
                        return 1
                z.extractall(tmp, members=names)
            sub = tmp / tops.pop()
        else:
            sub = path
        if not sub.is_dir():
            print(f"error: {sub} is not a folder")
            return 1
        sid = sub.name
        if sid in ("your_student_id", "example_student"):
            rep.warn(f"the folder is still named {sid!r}: rename it to your student ID")
        elif not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", sid):
            rep.error(f"folder name {sid!r}: use your student ID, lowercase letters and digits "
                      f"(it becomes your agents' names)")
        check_layout(sub, rep)
        for role in ("seller", "buyer"):
            if (sub / "agents").is_dir():
                check_role(sub / "agents" / role, f"agents/{role}", rep)
        print(f"submission {sid}:")
        preview(sub, rep)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    for w in rep.warnings:
        print(f"  warning: {w}")
    for e in rep.errors:
        print(f"  ERROR: {e}")
    print("OK: ready to hand in" if not rep.errors else f"{len(rep.errors)} error(s): fix them before handing in")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
