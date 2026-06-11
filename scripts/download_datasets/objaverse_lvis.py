import os
import shutil
import zipfile

from utils import (
    DATA_DIR,
    build_download_url,
    download_file,
    fetch_entries,
    file_sha256,
    get_expected_sha256,
    get_expected_size,
    parse_common_args,
    should_download,
)


DATASET_NAME = "objaverse_lvis"
API_TREE_BASE = "https://hf-mirror.com/api/datasets/auniquesun/Point-Cache/tree/main"
DOWNLOAD_BASES = [
    "https://hf-mirror.com/datasets/auniquesun/Point-Cache/resolve/main",
    "https://huggingface.co/datasets/auniquesun/Point-Cache/resolve/main",
]
ZIP_REPO_PATH = "objaverse_lvis.zip"
ZIP_PATH = os.path.join(DATA_DIR, ZIP_REPO_PATH)
TARGET_DIR = os.path.join(DATA_DIR, "objaverse_lvis")
MARKER_FILE = os.path.join(TARGET_DIR, "lvis_testset.txt")


def fetch_zip_entry():
    entries = fetch_entries(API_TREE_BASE, "")
    for entry in entries:
        if entry.get("type") == "file" and entry.get("path") == ZIP_REPO_PATH:
            return entry
    raise RuntimeError(f"远端文件列表中未找到: {ZIP_REPO_PATH}")


def download_zip(entry):
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    for download_base in DOWNLOAD_BASES:
        url = build_download_url(download_base, ZIP_REPO_PATH)
        print(f"下载地址: {url}")
        if download_file(url, ZIP_PATH):
            return True
        print("该下载源失败，尝试下一个下载源...")
    return False


def normalize_member_path(member_name):
    name = member_name.replace("\\", "/").lstrip("/")
    while name.startswith("./"):
        name = name[2:]

    if not name or name.endswith("/"):
        return None

    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts:
        return None

    if parts[0] == "objaverse_lvis":
        parts = parts[1:]

    if not parts:
        return None

    rel_path = os.path.normpath(os.path.join(*parts))
    if rel_path == ".." or rel_path.startswith(".." + os.sep):
        raise ValueError(f"zip 中存在不安全路径: {member_name}")

    return rel_path


def verify_zip_file(entry, check_sha256=True):
    print("\n开始校验 objaverse_lvis.zip...")
    if not os.path.exists(ZIP_PATH):
        print(f"缺失文件: {ZIP_PATH}")
        return False

    expected_size = get_expected_size(entry)
    actual_size = os.path.getsize(ZIP_PATH)
    if expected_size is not None and actual_size != expected_size:
        print(f"大小不一致: local={actual_size}, expected={expected_size}")
        return False

    if check_sha256:
        expected_sha256 = get_expected_sha256(entry)
        if expected_sha256 is None:
            print("远端缺少 SHA-256 元数据，跳过哈希校验。")
        else:
            actual_sha256 = file_sha256(ZIP_PATH)
            if actual_sha256 != expected_sha256:
                print(f"哈希不一致: local={actual_sha256}, expected={expected_sha256}")
                return False

    if not zipfile.is_zipfile(ZIP_PATH):
        print(f"不是有效 zip 文件: {ZIP_PATH}")
        return False

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"zip 内部 CRC 校验失败: {bad_file}")
            return False

    print("zip 校验通过。")
    return True


def extract_zip():
    os.makedirs(TARGET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = get_zip_members(zf)
        total = len(members)
        print(f"解压 {total} 个文件到: {TARGET_DIR}")

        for idx, (info, rel_path) in enumerate(members, start=1):
            dst_path = os.path.join(TARGET_DIR, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            with zf.open(info, "r") as src, open(dst_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if idx % 500 == 0 or idx == total:
                print(f"已解压 {idx}/{total}")


def get_zip_members(zf):
    members = []
    for info in zf.infolist():
        rel_path = normalize_member_path(info.filename)
        if rel_path is not None:
            members.append((info, rel_path))
    return members


def verify_extracted_files():
    if not os.path.isfile(MARKER_FILE):
        print(f"缺失解压标记文件: {MARKER_FILE}")
        return False

    missing = []
    size_mismatch = []
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = get_zip_members(zf)
        for info, rel_path in members:
            dst_path = os.path.join(TARGET_DIR, rel_path)
            if not os.path.isfile(dst_path):
                missing.append(rel_path)
                continue
            actual_size = os.path.getsize(dst_path)
            if actual_size != info.file_size:
                size_mismatch.append((rel_path, actual_size, info.file_size))

    if missing:
        print("\n解压目录缺失文件:")
        for rel_path in missing[:50]:
            print(f"- {rel_path}")
        if len(missing) > 50:
            print(f"... 还有 {len(missing) - 50} 个缺失文件")
        return False

    if size_mismatch:
        print("\n解压文件大小不一致:")
        for rel_path, actual_size, expected_size in size_mismatch[:50]:
            print(f"- {rel_path}: local={actual_size}, expected={expected_size}")
        if len(size_mismatch) > 50:
            print(f"... 还有 {len(size_mismatch) - 50} 个大小不一致文件")
        return False

    print(f"已找到解压标记文件: {MARKER_FILE}")
    print("解压目录校验通过。")
    return True


def main():
    args = parse_common_args(f"Download and verify {DATASET_NAME} files.")
    print(f"zip 文件路径: {ZIP_PATH}")
    print(f"解压目录: {TARGET_DIR}")

    try:
        entry = fetch_zip_entry()
    except Exception as exc:
        print(f"获取远端 zip 元数据失败: {exc}")
        raise SystemExit(1)

    if not args.verify_only:
        if should_download(entry, ZIP_PATH, force=args.force):
            if not download_zip(entry):
                print("所有下载源均失败。")
                raise SystemExit(1)
        else:
            print("跳过已完整 zip 文件。")

    if not verify_zip_file(entry, check_sha256=not args.no_sha256):
        raise SystemExit(1)

    if args.verify_only:
        if not verify_extracted_files():
            print("zip 文件正确，但解压目录不完整。请运行普通下载命令完成解压。")
            raise SystemExit(1)
        print("\nobjaverse_lvis 校验完成。")
        return

    if args.force and os.path.exists(TARGET_DIR):
        print(f"重新解压，删除旧目录: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    if not verify_extracted_files():
        extract_zip()
        if not verify_extracted_files():
            raise SystemExit(1)

    print("\nobjaverse_lvis 下载、校验并解压完成！")


if __name__ == "__main__":
    main()
