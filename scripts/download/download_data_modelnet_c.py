#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_download import download_hf_directory


def main():
    download_hf_directory(
        PROJECT_ROOT,
        repo_id='auniquesun/Point-PRC',
        repo_path='new-3ddg-benchmarks/xset/corruption/modelnet_c',
        local_name='modelnet_c',
    )


if __name__ == "__main__":
    main()
