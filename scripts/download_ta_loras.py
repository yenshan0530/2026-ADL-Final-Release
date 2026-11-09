"""Download the TA agents' LoRA weights (Qwen/Qwen3.5-4B) into agents/_ta/{seller,buyer}/checkpoint/.

The adapter configs are in the repository; the weights (about 466 MB each) are not, so
run this once before playing with the TA LLM agents:

    python scripts/download_ta_loras.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# folder -> Hugging Face repository
LORAS = {
    "agents/_ta/seller/checkpoint": "miulab/adl-qwen3.5-4b-seller-lora",
    "agents/_ta/buyer/checkpoint": "miulab/adl-qwen3.5-4b-buyer-lora",
}
FILES = ("adapter_config.json", "adapter_model.safetensors")


def remote_sha256(repo: str, name: str) -> str:
    """The sha256 Hugging Face reports for a large (LFS) file."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):     # the hash is on the first response
            return None
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{name}", method="HEAD")
    try:
        resp = urllib.request.build_opener(NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as redirect:          # 302 to the file store
        resp = redirect
    return (resp.headers.get("X-Linked-Etag") or "").strip('"')


def local_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, name: str, dest: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    tmp.replace(dest)
    print()


def main() -> int:
    for folder, repo in LORAS.items():
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        print(f"{repo} -> {folder}")
        for name in FILES:
            dest = out / name
            if name.endswith(".safetensors") and dest.exists() and \
                    local_sha256(dest) == remote_sha256(repo, name):
                print(f"  {name}: already there")           # the repository's current file
                continue
            fetch(repo, name, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
