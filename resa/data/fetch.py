"""Acquire every external resource ReSA needs.

1. IntelliSA hub + CLI repositories at the exact pinned commits (oracle labels,
   published baseline detections and rankings, Claude-labelled training items, vendored GLITCH).
2. The GLITCH ASE'22 replication package (figshare 19726603 v2, a 1.7 GB Docker image) and the
   extraction of its ``/RESULTS`` tree (oracle scripts + 196,755-script corpus + precomputed
   GLITCH corpus warnings) *without* needing Docker.

Usage:
    python -m resa.data.fetch all                 # everything
    python -m resa.data.fetch repos               # just the git repositories
    python -m resa.data.fetch glitch-package      # download + verify the tar
    python -m resa.data.fetch extract [--tar PATH] [--use-docker]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Set

import requests

from resa.config import Config, add_config_args, config_from_args
from resa.utils import ensure_dir, file_digest, get_logger, write_json

LOG = get_logger("resa.fetch")

KEEP_EXTENSIONS: Set[str] = {
    ".pp", ".rb", ".yml", ".yaml", ".csv", ".md", ".txt", ".sh", ".py", ".json", ".ini", ".erb", ".j2",
}


# ----------------------------------------------------------------------------- git
def _run(cmd: list, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    LOG.debug("$ %s", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def clone_pinned(url: str, commit: str, dest: Path) -> Path:
    """Clone ``url`` and check out ``commit`` (idempotent)."""
    if (dest / ".git").exists():
        head = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
        if head == commit:
            LOG.info("%s already at %s", dest.name, commit[:10])
            return dest
        LOG.info("%s at %s, moving to %s", dest.name, head[:10], commit[:10])
    else:
        ensure_dir(dest.parent)
        if dest.exists():
            shutil.rmtree(dest)
        _run(["git", "init", "-q", str(dest)])
        _run(["git", "remote", "add", "origin", url], cwd=dest)
    try:
        _run(["git", "fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
    except subprocess.CalledProcessError:
        LOG.warning("Shallow fetch of %s failed; fetching full history", commit[:10])
        _run(["git", "fetch", "-q", "origin"], cwd=dest)
        _run(["git", "checkout", "-q", commit], cwd=dest)
    head = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{url}: expected {commit}, got {head}")
    LOG.info("Cloned %s @ %s -> %s", url, commit[:10], dest)
    return dest


def fetch_repos(cfg: Config) -> Dict[str, Path]:
    ext = ensure_dir(cfg.path("external"))
    out = {}
    for key in ("intellisa_hub", "intellisa_cli"):
        src = cfg["sources"][key]
        out[key] = clone_pinned(src["url"], src["commit"], ext / key.replace("_", "-"))
    return out


# ----------------------------------------------------------------------------- figshare
def _figshare_file_url(article: int, version: int, filename: str) -> str:
    api = f"https://api.figshare.com/v2/articles/{article}/versions/{version}"
    resp = requests.get(api, timeout=60)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    for item in files:
        if item.get("name") == filename:
            return item["download_url"]
    if len(files) == 1:
        return files[0]["download_url"]
    raise RuntimeError(f"{filename} not found in figshare article {article} v{version}: {[f.get('name') for f in files]}")


def download_with_resume(url: str, dest: Path, chunk: int = 1 << 22, retries: int = 20) -> Path:
    part = Path(str(dest) + ".part")
    ensure_dir(dest.parent)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as resp:
                if resp.status_code == 416:  # already complete
                    break
                resp.raise_for_status()
                if have and resp.status_code != 206:
                    LOG.warning("Server ignored Range; restarting download")
                    have = 0
                    part.unlink(missing_ok=True)
                total = int(resp.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                written = have
                last = time.time()
                with open(part, mode) as handle:
                    for block in resp.iter_content(chunk_size=chunk):
                        if block:
                            handle.write(block)
                            written += len(block)
                            if time.time() - last > 10:
                                pct = 100.0 * written / total if total else 0.0
                                LOG.info("downloaded %.1f / %.1f MB (%.1f%%)", written / 1e6, total / 1e6, pct)
                                last = time.time()
            break
        except (requests.RequestException, OSError) as exc:
            wait = min(60, 2 ** attempt)
            LOG.warning("Download attempt %d failed (%s); retrying in %ss", attempt, exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Could not download {url}")
    os.replace(part, dest)
    return dest


def fetch_glitch_package(cfg: Config, tar_path: Optional[Path] = None) -> Path:
    src = cfg["sources"]["glitch_package"]
    raw = ensure_dir(cfg.path("raw"))
    dest = Path(tar_path) if tar_path else raw / src["filename"]
    if not dest.exists():
        url = _figshare_file_url(int(src["figshare_article"]), int(src["figshare_version"]), src["filename"])
        LOG.info("Downloading GLITCH replication package (%s) from %s", src["filename"], url)
        download_with_resume(url, dest)
    LOG.info("Verifying SHA1 of %s ...", dest)
    sha1 = file_digest(dest, "sha1")
    if sha1 != src["sha1"]:
        raise RuntimeError(
            f"SHA1 mismatch for {dest}: got {sha1}, expected {src['sha1']}. "
            "Delete the file and re-run, or download it manually from https://doi.org/10.6084/m9.figshare.19726603.v2"
        )
    LOG.info("SHA1 OK (%s)", sha1)
    return dest


# ----------------------------------------------------------------------------- extraction
def _safe_target(root: Path, member_name: str) -> Optional[Path]:
    rel = PurePosixPath(member_name.lstrip("./").lstrip("/"))
    if any(part == ".." for part in rel.parts):
        return None
    return root / Path(*rel.parts)


def _wanted(name: str, keep_all: bool) -> bool:
    if not name.startswith("RESULTS/"):
        return False
    if "/.git/" in name or name.endswith("/.git"):
        return False
    if keep_all:
        return True
    suffix = PurePosixPath(name).suffix.lower()
    return suffix in KEEP_EXTENSIONS


def _layer_names(outer: tarfile.TarFile) -> list:
    try:
        manifest = json.load(outer.extractfile("manifest.json"))  # docker save (legacy and OCI)
        return list(manifest[0]["Layers"])
    except KeyError:
        pass
    # Pure OCI layout: index.json -> manifest blob -> layers
    index = json.load(outer.extractfile("index.json"))
    digest = index["manifests"][0]["digest"].split(":", 1)[1]
    manifest = json.load(outer.extractfile(f"blobs/sha256/{digest}"))
    if "layers" not in manifest and "manifests" in manifest:  # nested index
        digest = manifest["manifests"][0]["digest"].split(":", 1)[1]
        manifest = json.load(outer.extractfile(f"blobs/sha256/{digest}"))
    return [f"blobs/sha256/{layer['digest'].split(':', 1)[1]}" for layer in manifest["layers"]]


def extract_results_from_image(tar_path: Path, out_root: Path, keep_all: bool = False) -> Dict[str, int]:
    """Replay the image's layers and materialise ``/RESULTS`` under ``out_root`` (no Docker needed)."""
    ensure_dir(out_root)
    stats = {"layers": 0, "files": 0, "whiteouts": 0, "skipped": 0, "links": 0}
    with tarfile.open(tar_path, "r") as outer:
        layers = _layer_names(outer)
        LOG.info("Image has %d layers", len(layers))
        for layer_name in layers:
            stats["layers"] += 1
            fileobj = outer.extractfile(layer_name)
            if fileobj is None:
                continue
            with tarfile.open(fileobj=fileobj, mode="r|*") as layer:
                for member in layer:
                    name = member.name.lstrip("./").lstrip("/")
                    if not name.startswith("RESULTS"):
                        continue
                    base = PurePosixPath(name).name
                    parent = str(PurePosixPath(name).parent)
                    if base == ".wh..wh..opq":  # opaque directory
                        target = _safe_target(out_root, parent)
                        if target and target.exists():
                            shutil.rmtree(target)
                        stats["whiteouts"] += 1
                        continue
                    if base.startswith(".wh."):
                        target = _safe_target(out_root, f"{parent}/{base[4:]}")
                        if target and target.exists():
                            shutil.rmtree(target) if target.is_dir() else target.unlink()
                        stats["whiteouts"] += 1
                        continue
                    if member.isdir():
                        continue
                    if not _wanted(name, keep_all):
                        stats["skipped"] += 1
                        continue
                    target = _safe_target(out_root, name)
                    if target is None:
                        continue
                    ensure_dir(target.parent)
                    if member.isreg():
                        src = layer.extractfile(member)
                        if src is None:
                            continue
                        with open(target, "wb") as handle:
                            shutil.copyfileobj(src, handle)
                        stats["files"] += 1
                    elif member.islnk():
                        link_src = _safe_target(out_root, member.linkname.lstrip("./").lstrip("/"))
                        if link_src and link_src.exists():
                            shutil.copyfile(link_src, target)
                            stats["links"] += 1
                    elif member.issym():
                        link = member.linkname
                        resolved = (target.parent / link) if not link.startswith("/") else _safe_target(out_root, link)
                        if resolved and resolved.exists() and resolved.is_file():
                            shutil.copyfile(resolved, target)
                            stats["links"] += 1
            LOG.info("layer %d/%d done (%d files so far)", stats["layers"], len(layers), stats["files"])
    return stats


def extract_with_docker(tar_path: Path, out_root: Path) -> Dict[str, int]:
    """Alternative extraction through a local Docker daemon."""
    _run(["docker", "load", "-i", str(tar_path)])
    cid = _run(["docker", "create", "glitch"]).stdout.strip()
    try:
        ensure_dir(out_root)
        _run(["docker", "cp", f"{cid}:/RESULTS", str(out_root)])
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)
    count = sum(1 for _ in (out_root / "RESULTS").rglob("*") if _.is_file())
    return {"files": count}


def extract(cfg: Config, tar_path: Optional[Path] = None, use_docker: bool = False, keep_all: bool = False) -> Path:
    raw = cfg.path("raw")
    tar_path = Path(tar_path) if tar_path else raw / cfg["sources"]["glitch_package"]["filename"]
    if not tar_path.exists():
        raise FileNotFoundError(f"{tar_path} missing; run `python -m resa.data.fetch glitch-package` first")
    out_root = raw / "glitch_package"
    marker = out_root / ".extracted.json"
    if marker.exists():
        LOG.info("Already extracted to %s", out_root)
        return out_root / "RESULTS"
    stats = extract_with_docker(tar_path, out_root) if use_docker else extract_results_from_image(tar_path, out_root, keep_all)
    if not (out_root / "RESULTS").exists():
        raise RuntimeError("Extraction finished but no RESULTS directory was produced")
    write_json(marker, stats)
    LOG.info("Extracted RESULTS: %s", stats)
    return out_root / "RESULTS"


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["all", "repos", "glitch-package", "extract"])
    parser.add_argument("--tar", type=Path, default=None, help="Use an already-downloaded glitch-latest.tar")
    parser.add_argument("--use-docker", action="store_true", help="Extract via a Docker daemon instead of pure Python")
    parser.add_argument("--keep-all", action="store_true", help="Extract every file under RESULTS, not only text/IaC")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if args.what in ("all", "repos"):
        fetch_repos(cfg)
    if args.what in ("all", "glitch-package"):
        fetch_glitch_package(cfg, args.tar)
    if args.what in ("all", "extract"):
        extract(cfg, args.tar, args.use_docker, args.keep_all)


if __name__ == "__main__":
    main()
