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


def download_text_encoder():
    print("\n=== Download Uni3D text encoder ===")
    text_dir = os.path.join(WEIGHTS_DIR, "uni3d", "open_clip_pytorch_model")
    ensure_dir(text_dir)

    text_src = os.path.join(text_dir, "open_clip_pytorch_model.bin")
    if not os.path.isfile(text_src):
        ok = download_with_fallback(
            "timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/resolve/main/open_clip_pytorch_model.bin",
            text_src,
        )
        if not ok:
            raise SystemExit("Failed to download Uni3D text encoder file")

    # Code expects this exact filename.
    text_alias = os.path.join(text_dir, "laion2b_s9b_b144k.bin")
    ensure_alias(text_src, text_alias)


def download_point_encoder_main():
    print("\n=== Download Uni3D point encoder (main) ===")
    point_dir = os.path.join(WEIGHTS_DIR, "uni3d", "pc_encoder")
    ensure_dir(point_dir)

    main_ckpt = os.path.join(point_dir, "uni3d_g_ensembled_model.pt")
    if not os.path.isfile(main_ckpt):
        ok = download_with_fallback(
            "BAAI/Uni3D/resolve/main/modelzoo/uni3d-g/model.pt",
            main_ckpt,
        )
        if not ok:
            raise SystemExit("Failed to download Uni3D main checkpoint")


def download_task_specific_ckpts():
    print("\n=== Download Uni3D task-specific checkpoints ===")
    mapping = {
        "lvis": "modelzoo/uni3d-g/lvis/model.pt",
        "modelnet40": "modelzoo/uni3d-g/mnet40/model.pt",
        "scanobjnn": "modelzoo/uni3d-g/scanobjnn/model.pt",
    }

    for name, repo_rel in mapping.items():
        target_dir = os.path.join(WEIGHTS_DIR, "uni3d", name)
        ensure_dir(target_dir)
        target_file = os.path.join(target_dir, "model.pt")

        if os.path.isfile(target_file):
            print(f"Skip existing: {target_file}")
            continue

        ok = download_with_fallback(f"BAAI/Uni3D/resolve/main/{repo_rel}", target_file)
        if not ok:
            raise SystemExit(f"Failed to download Uni3D task checkpoint: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download Uni3D weights required by Point-Cache")
    parser.add_argument(
        "--with-task-ckpts",
        action="store_true",
        help="also download task-specific checkpoints under weights/uni3d/{lvis,modelnet40,scanobjnn}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    download_text_encoder()
    download_point_encoder_main()

    if args.with_task_ckpts:
        download_task_specific_ckpts()

    print("\nUni3D weights download completed.")


if __name__ == "__main__":
    main()
