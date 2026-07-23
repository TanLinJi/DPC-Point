"""Utilities for downloading DPC-Point datasets.

The helpers in this module use paths relative to the current DPC-Point
repository. They are intentionally limited to dataset downloads; model weights
should use a separate utility module.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote, urljoin

USER_AGENT = "Mozilla/5.0"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def data_dir(project_root: Path, relative_dir: str = "data") -> Path:
    target = project_root / relative_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_repo_path(path: str) -> str:
    return str(path).replace("\\", "/").strip().strip("/")


def safe_relative_path(path: str) -> Path:
    clean = normalize_repo_path(path)
    parts = [part for part in clean.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe relative path: {path}")
    return Path(*parts)


def local_relative_path(full_path: str, root_path: str) -> Path:
    full = normalize_repo_path(full_path)
    root = normalize_repo_path(root_path)
    prefix = f"{root}/" if root else ""

    if prefix and full.startswith(prefix):
        rel = full[len(prefix) :]
    elif root and full == root:
        raise ValueError(f"Expected a file below {root_path}, got the root path itself")
    elif "/" in full:
        rel = full.split("/", 1)[1]
    else:
        rel = full

    return safe_relative_path(rel)


def hf_tree_url(repo_id: str, repo_path: str, revision: str = "main") -> str:
    path = normalize_repo_path(repo_path)
    base = f"{hf_endpoint()}/api/datasets/{repo_id}/tree/{revision}"
    if not path:
        return base
    return f"{base}/{quote(path, safe='/')}"


def hf_file_url(repo_id: str, repo_path: str, revision: str = "main") -> str:
    path = normalize_repo_path(repo_path)
    return f"{hf_endpoint()}/datasets/{repo_id}/resolve/{revision}/{quote(path, safe='/')}"


def open_url(url: str, timeout: int = 30, max_redirects: int = 5):
    current_url = url
    for _ in range(max_redirects + 1):
        req = urllib.request.Request(current_url, headers={"User-Agent": USER_AGENT})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location")
            if exc.code in REDIRECT_STATUS_CODES and location:
                next_url = urljoin(current_url, location)
                print(f"Redirect {exc.code}: {current_url} -> {next_url}")
                current_url = next_url
                continue
            raise

    raise RuntimeError(f"Too many redirects while requesting: {url}")


def request_json(url: str, timeout: int = 30):
    with open_url(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hf_entries(repo_id: str, repo_path: str, revision: str = "main"):
    url = hf_tree_url(repo_id, repo_path, revision)
    return request_json(url)


def iter_hf_files(repo_id: str, root_path: str, revision: str = "main"):
    stack = [normalize_repo_path(root_path)]
    while stack:
        current = stack.pop()
        print(f"Scanning: {current}")
        entries = fetch_hf_entries(repo_id, current, revision)
        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not entry_path:
                continue
            if entry_type in ("directory", "dir"):
                stack.append(normalize_repo_path(entry_path))
            elif entry_type == "file":
                yield normalize_repo_path(entry_path)


def run_wget(url: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "--show-progress", url, "-O", str(target)]
    return subprocess.run(cmd, check=False).returncode == 0


def download_with_urllib(url: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url, timeout=60) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return True
    except Exception as exc:
        print(f"urllib download failed: {exc}")
        return False


def download_file(url: str, target: Path) -> bool:
    if shutil.which("wget"):
        return run_wget(url, target)
    print("wget is not available; falling back to urllib without resume support.")
    return download_with_urllib(url, target)


def download_hf_directory(
    project_root: Path,
    *,
    repo_id: str,
    repo_path: str,
    local_name: str,
    revision: str = "main",
    data_relative_dir: str = "data",
) -> Path:
    project_root = project_root.resolve()
    target_dir = data_dir(project_root, data_relative_dir) / local_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Dataset: {local_name}")
    print(f"Repository: {repo_id}")
    print(f"Repository path: {repo_path}")
    print(f"HF endpoint: {hf_endpoint()}")
    print(f"Target directory: {target_dir.relative_to(project_root)}")
    print("=" * 72)

    try:
        files = sorted(iter_hf_files(repo_id, repo_path, revision))
    except Exception as exc:
        raise SystemExit(f"Failed to fetch Hugging Face file tree: {exc}") from exc

    print(f"Found {len(files)} files. Starting download...")

    failed = []
    for index, full_path in enumerate(files, start=1):
        try:
            rel_path = local_relative_path(full_path, repo_path)
        except ValueError as exc:
            failed.append((full_path, str(exc)))
            continue

        url = hf_file_url(repo_id, full_path, revision)
        target = target_dir / rel_path
        print(f"\n[{index}/{len(files)}] {rel_path}")
        if not download_file(url, target):
            failed.append((str(rel_path), url))

    if failed:
        print("\nThe following files failed to download:")
        for item, reason in failed:
            print(f"- {item} -> {reason}")
        raise SystemExit(1)

    print(f"\n{local_name} download completed: {target_dir.relative_to(project_root)}")
    return target_dir


def normalize_zip_member(member_name: str, strip_top_level: str | None = None) -> Path | None:
    name = member_name.replace("\\", "/").lstrip("/")
    while name.startswith("./"):
        name = name[2:]
    if not name or name.endswith("/"):
        return None

    parts = [part for part in name.split("/") if part not in ("", ".")]
    if strip_top_level and parts and parts[0] == strip_top_level:
        parts = parts[1:]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe path in zip: {member_name}")
    return Path(*parts)


def extract_zip(zip_path: Path, target_dir: Path, strip_top_level: str | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = []
        for info in archive.infolist():
            rel_path = normalize_zip_member(info.filename, strip_top_level)
            if rel_path is not None:
                members.append((info, rel_path))

        for index, (info, rel_path) in enumerate(members, start=1):
            target = target_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if index % 500 == 0 or index == len(members):
                print(f"Extracted {index}/{len(members)}")
