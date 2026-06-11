import os

from utils import DATA_DIR, download_dataset


DATASET_NAME = "modelnet_c"
API_TREE_BASE = "https://hf-mirror.com/api/datasets/auniquesun/Point-PRC/tree/main"
DOWNLOAD_BASES = [
    "https://hf-mirror.com/datasets/auniquesun/Point-PRC/resolve/main",
    "https://huggingface.co/datasets/auniquesun/Point-PRC/resolve/main",
]
ROOT_PATH = "new-3ddg-benchmarks/xset/corruption/modelnet_c"
SAVE_DIR = os.path.join(DATA_DIR, "modelnet_c")


def main():
    download_dataset(
        dataset_name=DATASET_NAME,
        api_tree_base=API_TREE_BASE,
        download_bases=DOWNLOAD_BASES,
        root_path=ROOT_PATH,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    main()
