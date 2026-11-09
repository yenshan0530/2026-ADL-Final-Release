"""Local vLLM servers: launch base-model replicas on GPUs, register LoRA adapters at runtime.

One server per replica holds the base model once. Every student adapter is a few hundred
MB at most, and vLLM serves many of them from one base: requests name the adapter as their
`model`, and one batch can mix requests for different adapters. So the whole class runs
on one base-model copy per GPU instead of one model per student.

Adapters are registered through vLLM's runtime API (POST /v1/load_lora_adapter, enabled by
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True), so nothing has to be listed at startup. Each
adapter goes to the one replica its name hashes to (`ModelSpec.url_for`), the same one
its requests are routed to.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from adlmarket.config import LoadedRun, ModelSpec, ServingSpec


def _http(method: str, url: str, body: Optional[dict] = None, key: str = "EMPTY",
          timeout: float = 30.0) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def served_models(url: str, key: str = "EMPTY") -> Optional[dict[str, Optional[str]]]:
    """Model names a server answers to, with the path each LoRA was loaded from (`root`;
    the base model's is its name), or None if the server is not reachable."""
    try:
        status, text = _http("GET", f"{url}/models", key=key, timeout=5)
    except (urllib.error.URLError, OSError):
        return None
    if status != 200:
        return None
    return {m["id"]: m.get("root") for m in json.loads(text).get("data", [])}


def vllm_command(model: ModelSpec, sv: ServingSpec, replica: int) -> tuple[list[str], dict[str, str]]:
    """argv and extra environment for one replica."""
    gpus = str(sv.gpus[replica])
    argv = [*sv.command, model.model,
            "--host", sv.host, "--port", str(sv.port + replica),
            "--enable-lora", "--max-lora-rank", str(sv.max_lora_rank),
            "--max-loras", str(sv.max_loras), "--max-cpu-loras", str(sv.max_cpu_loras),
            "--gpu-memory-utilization", str(sv.gpu_memory_utilization),
            "--max-model-len", str(sv.max_model_len),
            "--enable-auto-tool-choice", "--tool-call-parser", sv.tool_call_parser]
    if "," in gpus:
        argv += ["--tensor-parallel-size", str(len(gpus.split(",")))]
    argv += sv.extra_args
    # PCI_BUS_ID: GPU numbers mean what nvidia-smi shows. CUDA's default order puts the
    # fastest card first, so on a mixed machine "4" could be a different GPU.
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpus,
           "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    # vLLM compiles some kernels on first start (Qwen3.5's linear attention) and needs the
    # tools of its own environment (ninja): put the folder of the vllm executable first
    exe = shutil.which(sv.command[0]) if sv.command else None
    if exe:
        env["PATH"] = os.pathsep.join([str(Path(exe).resolve().parent), os.environ.get("PATH", "")])
    return argv, env


class VLLMCluster:
    """Start every replica in `run.serving`, wait until all answer, stop them on exit."""

    def __init__(self, run: LoadedRun, log_dir: Path):
        self.run = run
        self.log_dir = log_dir
        self.procs: list[tuple[str, subprocess.Popen, Path]] = []

    def __enter__(self) -> "VLLMCluster":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for name, sv in self.run.serving.items():
                for i, url in enumerate(sv.urls()):
                    if served_models(url) is not None:
                        raise RuntimeError(f"{url} is already serving; stop it or omit --launch-vllm")
                    argv, env = vllm_command(self.run.models[name], sv, i)
                    log = self.log_dir / f"vllm_{name}_{i}.log"
                    proc = subprocess.Popen(argv, env={**os.environ, **env},
                                            stdout=log.open("w"), stderr=subprocess.STDOUT)
                    self.procs.append((url, proc, log))
                    print(f"launched {name} replica {i} on GPU {sv.gpus[i]} -> {url} (log {log})")
            for name, sv in self.run.serving.items():
                for url in sv.urls():
                    self._wait(url, sv.startup_timeout)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait(self, url: str, timeout: float) -> None:
        proc, log = next((p, l) for u, p, l in self.procs if u == url)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = log.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"vLLM for {url} exited with {proc.returncode}:\n{tail}")
            if served_models(url) is not None:
                print(f"ready: {url}")
                return
            time.sleep(2)
        raise RuntimeError(f"vLLM for {url} not ready after {timeout:.0f}s (see {log})")

    def __exit__(self, *exc: Any) -> None:
        for _, proc, _ in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs = []


def rename_adapter(src: Path, rename: dict[str, str], dest: Path) -> tuple[Path, int]:
    """A copy of the LoRA in `src` with its weight names renamed ({old: new}, first match
    in each name, skipped where `new` is already there), written to `dest`. Only the
    safetensors header (a JSON index of names and byte ranges) changes; the tensor bytes
    are copied as they are. Returns the folder to load and how many names changed."""
    weights = src / "adapter_model.safetensors"
    if not weights.exists():
        raise ValueError(f"{src}: lora_rename needs adapter_model.safetensors")
    raw = weights.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    out, changed = {}, 0
    for key, info in header.items():
        new_key = key
        if key != "__metadata__":
            for old, repl in rename.items():
                if old in key and repl not in key:
                    new_key = key.replace(old, repl, 1)
                    break
        changed += new_key != key
        out[new_key] = info
    if not changed:
        return src, 0
    blob = json.dumps(out, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)                       # keep the data 8-byte aligned
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dest / f.name)
    tmp = dest / "adapter_model.safetensors.tmp"
    with tmp.open("wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        fh.write(raw[8 + n:])
    tmp.replace(dest / "adapter_model.safetensors")
    return dest, changed


def adapter_fingerprint(src: Path, rename: Optional[dict[str, str]] = None) -> str:
    """A short key that changes whenever the adapter is retrained: its config, the weights'
    size and modification time, their first and last MB, and the rename rule. Cheap even
    for hundreds of adapters (no full read)."""
    h = hashlib.sha256()
    h.update((src / "adapter_config.json").read_bytes())
    w = src / "adapter_model.safetensors"
    if w.exists():
        st = w.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        with w.open("rb") as f:
            h.update(f.read(1 << 20))
            f.seek(max(0, st.st_size - (1 << 20)))
            h.update(f.read(1 << 20))
    h.update(json.dumps(rename or {}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def cached_adapter(name: str, src: Path, rename: Optional[dict[str, str]], cache: Path) -> Path:
    """The folder vLLM loads for this adapter: `<cache>/<name>-<fingerprint>`, shared by all
    runs. With `rename` and weight names to change, a renamed copy (written once per
    version of the adapter); otherwise a link to the adapter itself (no copy)."""
    src = Path(src).resolve()
    dest = Path(cache) / f"{name}-{adapter_fingerprint(src, rename)}"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for old in dest.parent.glob(f"{name}-*"):            # earlier versions of this adapter
        if old.is_symlink() or not old.is_dir():
            old.unlink(missing_ok=True)
        else:
            shutil.rmtree(old, ignore_errors=True)
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    out, changed = (rename_adapter(src, rename, tmp) if rename else (src, 0))
    try:
        if changed:
            tmp.rename(dest)
        else:
            dest.symlink_to(src, target_is_directory=True)
    except FileExistsError:                              # another run made it meanwhile
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def default_adapter_cache(run: LoadedRun) -> Path:
    """Where renamed adapters are kept between runs: <repo>/.cache/adapters (git-ignored)."""
    return Path(run.root) / ".cache" / "adapters"


def register_adapters(run: LoadedRun, cache: Optional[Path] = None) -> int:
    """Make sure every LoRA in the run is loaded, in its current version, on the replica its
    requests go to. Returns how many were (re)loaded; safe to call repeatedly. Adapters are
    loaded from the shared cache (cached_adapter): renamed copies with `lora_rename`, links
    otherwise. An adapter already on the server from another path (an older version, e.g.
    retrained since a kept-up server loaded it) is unloaded and loaded again."""
    cache = Path(cache) if cache is not None else default_adapter_cache(run)
    loaded = 0
    for name, m in sorted(run.models.items()):
        if not m.lora:
            continue
        url = m.url_for(m.model)
        key = m.resolve_key()
        have = served_models(url, key)
        if have is None:
            raise RuntimeError(f"model server {url} is not reachable (needed for {name}); "
                               f"start it, or pass --launch-vllm")
        path = str(cached_adapter(name, Path(m.lora.path), m.lora_rename or None, cache))
        if name in have:
            if have[name] == path:
                continue                                 # this version is already loaded
            _http("POST", f"{url.removesuffix('/v1')}/v1/unload_lora_adapter",
                  {"lora_name": name}, key=key, timeout=60)
        status, text = _http("POST", f"{url.removesuffix('/v1')}/v1/load_lora_adapter",
                             {"lora_name": name, "lora_path": path}, key=key, timeout=300)
        if status == 200:
            loaded += 1
            continue
        if status == 404:
            raise RuntimeError(f"{url} does not allow runtime LoRA loading: start vLLM with "
                               f"VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and --enable-lora")
        raise RuntimeError(f"loading LoRA {name} on {url} failed ({status}): {text[:500]}")
    return loaded
