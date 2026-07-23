import argparse
import os
import shutil
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = SCRIPT_DIR


def download_with_fallback(relative_repo_path, save_path):
    mirror_url = f"https://hf-mirror.com/{relative_repo_path}"
    official_url = f"https://huggingface.co/{relative_repo_path}"

    for url in (mirror_url, official_url):
        print(f"Trying: {url}")
        cmd = ["wget", "-c", "--show-progress", url, "-O", save_path]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("Download succeeded.")
            return True
        print("Download failed on this source, trying next source...")

    return False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_alias(src_file, alias_file):
    if os.path.exists(alias_file):
        return

    try:
        os.symlink(os.path.basename(src_file), alias_file)
        print(f"Created symlink: {alias_file} -> {os.path.basename(src_file)}")
        return
    except OSError:
        pass

    try:
        os.link(src_file, alias_file)
        print(f"Created hardlink: {alias_file}")
        return
    except OSError:
        pass

    shutil.copy2(src_file, alias_file)
    print(f"Created file copy: {alias_file}")


def maybe_reuse_legacy_model(target_model_path):
    legacy = os.path.join(WEIGHTS_DIR, "openshape", "model.pt")
    if os.path.isfile(legacy) and not os.path.isfile(target_model_path):
        ensure_dir(os.path.dirname(target_model_path))
        shutil.copy2(legacy, target_model_path)
        print(f"Reused existing file: {legacy} -> {target_model_path}")


def download_vitg14():
    print("\n=== Download OpenShape vitg14 weights ===")

    model_dir = os.path.join(WEIGHTS_DIR, "openshape", "openshape-pointbert-vitg14-rgb")
    text_dir = os.path.join(WEIGHTS_DIR, "openshape", "open_clip_pytorch_model", "vit-bigG-14")

    ensure_dir(model_dir)
    ensure_dir(text_dir)

    model_path = os.path.join(model_dir, "model.pt")
    maybe_reuse_legacy_model(model_path)
    if not os.path.isfile(model_path):
        ok = download_with_fallback(
            "OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt",
            model_path,
        )
        if not ok:
            raise SystemExit("Failed to download OpenShape vitg14 point encoder model.pt")

    text_src = os.path.join(text_dir, "open_clip_pytorch_model.bin")
    if not os.path.isfile(text_src):
        ok = download_with_fallback(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin",
            text_src,
        )
        if not ok:
            raise SystemExit("Failed to download OpenShape vitg14 text encoder file")

    # Code expects this exact filename.
    text_alias = os.path.join(text_dir, "laion2b_s39b_b160k.bin")
    ensure_alias(text_src, text_alias)


def download_vitl14():
    print("\n=== Download OpenShape vitl14 weights ===")

    model_dir = os.path.join(WEIGHTS_DIR, "openshape", "openshape-pointbert-vitl14-rgb")
    text_dir = os.path.join(WEIGHTS_DIR, "openshape", "open_clip_pytorch_model", "vit-l-14")

    ensure_dir(model_dir)
    ensure_dir(text_dir)

    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.isfile(model_path):
        ok = download_with_fallback(
            "OpenShape/openshape-pointbert-vitl14-rgb/resolve/main/model.pt",
            model_path,
        )
        if not ok:
            raise SystemExit("Failed to download OpenShape vitl14 point encoder model.pt")

    text_src = os.path.join(text_dir, "open_clip_pytorch_model.bin")
    if not os.path.isfile(text_src):
        ok = download_with_fallback(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin",
            text_src,
        )
        if not ok:
            raise SystemExit("Failed to download OpenShape vitl14 text encoder file")

    # Code expects this exact filename.
    text_alias = os.path.join(text_dir, "laion2b_s32b_b82k.bin")
    ensure_alias(text_src, text_alias)


def parse_args():
    parser = argparse.ArgumentParser(description="Download OpenShape weights required by Point-Cache")
    parser.add_argument(
        "--variant",
        choices=["vitg14", "vitl14", "both"],
        default="vitg14",
        help="which OpenShape variant to download (default: vitg14)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.variant in ("vitg14", "both"):
        download_vitg14()

    if args.variant in ("vitl14", "both"):
        download_vitl14()

    print("\nOpenShape weights download completed.")


if __name__ == "__main__":
    main()
