import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

REDIRECT_CODES = (301, 302, 303, 307, 308)


def normalize_repo_path(path):
    return str(path).replace("\\", "/").strip().lstrip("/")


def build_api_url(api_tree_base, repo_path):
    clean_path = normalize_repo_path(repo_path)
    base = api_tree_base.rstrip("/")
    if not clean_path:
        return base
    return f"{base}/{urllib.parse.quote(clean_path, safe='/')}"


def fetch_json(url, timeout=20, max_redirects=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    current_url = url
    for _ in range(max_redirects + 1):
        req = urllib.request.Request(current_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in REDIRECT_CODES:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            current_url = urllib.parse.urljoin(current_url, location)
            print(f"目录 API 重定向到: {current_url}")
    raise RuntimeError(f"目录 API 重定向次数超过限制: {max_redirects}")


def fetch_entries(api_tree_base, repo_path):
    return fetch_json(build_api_url(api_tree_base, repo_path))


def collect_file_entries(api_tree_base, root_path, recursive=False):
    root_path = normalize_repo_path(root_path)
    file_entries = []
    stack = [root_path]

    while stack:
        current = stack.pop()
        print(f"扫描目录: {current or '<repo-root>'}")
        entries = fetch_entries(api_tree_base, current)

        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not entry_path:
                continue

            if entry_type == "file":
                file_entries.append(entry)
            elif recursive and entry_type in ("directory", "dir"):
                stack.append(normalize_repo_path(entry_path))

    return sorted(file_entries, key=lambda item: item.get("path", ""))


def to_local_relative_path(full_path, root_path):
    clean_full = normalize_repo_path(full_path)
    clean_root = normalize_repo_path(root_path).rstrip("/")

    if clean_root:
        prefix = clean_root + "/"
        if clean_full.startswith(prefix):
            rel_path = clean_full[len(prefix) :]
        elif prefix in clean_full:
            rel_path = clean_full.split(prefix, 1)[1]
        else:
            rel_path = os.path.basename(clean_full)
    else:
        rel_path = clean_full

    rel_path = normalize_repo_path(rel_path)
    parts = rel_path.split("/") if rel_path else []
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"非法相对路径: full_path={full_path}, root_path={root_path}")
    return os.path.normpath(os.path.join(*parts))


def build_download_url(download_base, full_path):
    clean_full = normalize_repo_path(full_path)
    return f"{download_base.rstrip('/')}/{urllib.parse.quote(clean_full, safe='/')}"


def get_expected_size(entry):
    if entry.get("lfs") and entry["lfs"].get("size") is not None:
        return int(entry["lfs"]["size"])
    if entry.get("size") is not None:
        return int(entry["size"])
    return None


def get_expected_sha256(entry):
    if entry.get("lfs") and entry["lfs"].get("oid"):
        return entry["lfs"]["oid"]
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_download(entry, save_path, force=False):
    if force:
        return True
    if not os.path.exists(save_path):
        return True
    expected_size = get_expected_size(entry)
    if expected_size is None:
        return False
    return os.path.getsize(save_path) != expected_size


def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd = ["wget", "-c", "--show-progress", url, "-O", save_path]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def download_with_fallbacks(download_bases, full_path, save_path):
    for download_base in download_bases:
        url = build_download_url(download_base, full_path)
        print(f"下载地址: {url}")
        if download_file(url, save_path):
            return True
        print("该下载源失败，尝试下一个下载源...")
    return False


def verify_entries(file_entries, save_dir, root_path, check_sha256=True):
    missing = []
    size_mismatch = []
    hash_mismatch = []
    hash_unavailable = []

    print("\n开始校验数据集完整性...")
    for entry in file_entries:
        rel_path = to_local_relative_path(entry["path"], root_path)
        save_path = os.path.join(save_dir, rel_path)
        expected_size = get_expected_size(entry)
        expected_sha256 = get_expected_sha256(entry)

        if not os.path.exists(save_path):
            missing.append(rel_path)
            continue

        actual_size = os.path.getsize(save_path)
        if expected_size is not None and actual_size != expected_size:
            size_mismatch.append((rel_path, actual_size, expected_size))
            continue

        if check_sha256:
            if expected_sha256 is None:
                hash_unavailable.append(rel_path)
                continue
            actual_sha256 = file_sha256(save_path)
            if actual_sha256 != expected_sha256:
                hash_mismatch.append((rel_path, actual_sha256, expected_sha256))

    print(f"远端文件数: {len(file_entries)}")
    print(f"缺失文件数: {len(missing)}")
    print(f"大小不一致文件数: {len(size_mismatch)}")
    if check_sha256:
        print(f"哈希不一致文件数: {len(hash_mismatch)}")
        if hash_unavailable:
            print(f"缺少远端 SHA-256 元数据文件数: {len(hash_unavailable)}")

    if missing:
        print("\n缺失文件:")
        for rel_path in missing:
            print(f"- {rel_path}")

    if size_mismatch:
        print("\n大小不一致文件:")
        for rel_path, actual_size, expected_size in size_mismatch:
            print(f"- {rel_path}: local={actual_size}, expected={expected_size}")

    if hash_mismatch:
        print("\n哈希不一致文件:")
        for rel_path, actual_sha256, expected_sha256 in hash_mismatch:
            print(f"- {rel_path}: local={actual_sha256}, expected={expected_sha256}")

    ok = not missing and not size_mismatch and not hash_mismatch
    if ok:
        if check_sha256:
            print("\n校验通过：文件数量、文件大小、SHA-256 内容哈希均匹配。")
        else:
            print("\n校验通过：文件数量和文件大小均匹配。")
    else:
        print("\n校验失败：请重新运行下载脚本补齐或修复上述文件。")
    return ok


def parse_common_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只校验本地文件，不下载。适合下载完成后单独检查。",
    )
    parser.add_argument(
        "--no-sha256",
        action="store_true",
        help="只做文件数量和大小校验，跳过 SHA-256 内容哈希校验。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新下载所有文件。",
    )
    return parser.parse_args()


def download_dataset(
    dataset_name,
    api_tree_base,
    download_bases,
    root_path,
    save_dir,
    recursive=False,
):
    args = parse_common_args(f"Download and verify {dataset_name} files.")
    if isinstance(download_bases, str):
        download_bases = [download_bases]

    os.makedirs(save_dir, exist_ok=True)
    print(f"数据将下载到: {save_dir}")
    print(f"开始扫描远端文件树: {root_path}")

    try:
        file_entries = collect_file_entries(api_tree_base, root_path, recursive=recursive)
    except Exception as exc:
        print(f"获取目录树失败，请检查网络或地址: {exc}")
        raise SystemExit(1)

    print(f"成功获取目录，共发现 {len(file_entries)} 个文件。")

    if not args.verify_only:
        failed = []
        skipped = 0
        print("开始下载/补齐文件...")

        for entry in file_entries:
            rel_path = to_local_relative_path(entry["path"], root_path)
            save_path = os.path.join(save_dir, rel_path)

            if not should_download(entry, save_path, force=args.force):
                skipped += 1
                print(f"\n>>> 跳过已完整文件: {rel_path}")
                continue

            print(f"\n>>> 正在处理: {rel_path}")
            if not download_with_fallbacks(download_bases, entry["path"], save_path):
                failed.append(rel_path)

        print(f"\n跳过已完整文件数: {skipped}")
        if failed:
            print("\n以下文件下载失败：")
            for rel_path in failed:
                print(f"- {rel_path}")
            raise SystemExit(1)

    if not verify_entries(file_entries, save_dir, root_path, check_sha256=not args.no_sha256):
        raise SystemExit(1)

    print(f"\n{dataset_name} 下载并校验完成！")
