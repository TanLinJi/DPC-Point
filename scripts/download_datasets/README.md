# Dataset Download Scripts

This directory stores dataset download and verification scripts for DPC-Point.

Users can either manually download datasets from the HuggingFace repositories listed in the project-level `README.md`, or use the scripts here. The scripts are preferred for reproducibility because they automatically verify file count, file size, and SHA-256 checksum when remote metadata is available.

## Reproducible Paths

Each script resolves paths from its own location, not from the current working directory. After cloning DPC-Point to any local path, downloaded files are saved to:

```text
<DPC-Point>/data/<dataset-name>
```

For example, both commands below are valid:

```bash
cd <DPC-Point>
python scripts/download_datasets/modelnet40_c.py
```

```bash
cd /tmp
python /path/to/DPC-Point/scripts/download_datasets/modelnet40_c.py
```

In both cases, the output directory is still:

```text
<DPC-Point>/data/modelnet40_c
```

## Common Usage

All dataset scripts support the same common options.

Download or resume:

```bash
python scripts/download_datasets/<script>.py
```

The script will request the remote file list, download missing or incomplete files, skip complete files, and verify the dataset after downloading.

Verify only:

```bash
python scripts/download_datasets/<script>.py --verify-only
```

This does not download files. It checks whether the local dataset is complete and correct.

Fast verification:

```bash
python scripts/download_datasets/<script>.py --verify-only --no-sha256
```

This checks file count and file size but skips SHA-256 checksum verification. Use it only for quick intermediate checks. Before reporting or reproducing paper results, use the full verification command without `--no-sha256`.

Force redownload:

```bash
python scripts/download_datasets/<script>.py --force
```

This redownloads all files and is useful when local files may be corrupted.

## Dataset Commands

| Dataset | Script | Output directory |
| --- | --- | --- |
| ModelNet-C | `modelnet_c.py` | `<DPC-Point>/data/modelnet_c` |
| ModelNet40-C | `modelnet40_c.py` | `<DPC-Point>/data/modelnet40_c` |
| ShapeNet-C | `shapenet_c.py` | `<DPC-Point>/data/shapenet_c` |
| ScanObjectNN-C | `scanobjectnn_c.py` | `<DPC-Point>/data/sonn_c` |
| OmniObject3D | `omniobject3d.py` | `<DPC-Point>/data/omniobject3d` |
| ModelNet40 | `modelnet40.py` | `<DPC-Point>/data/modelnet40` |
| ScanObjectNN | `scanobjectnn.py` | `<DPC-Point>/data/scanobjnn` |
| Objaverse-LVIS | `objaverse_lvis.py` | `<DPC-Point>/data/objaverse_lvis` |

Examples:

```bash
python scripts/download_datasets/modelnet_c.py
python scripts/download_datasets/modelnet40_c.py
python scripts/download_datasets/shapenet_c.py
python scripts/download_datasets/scanobjectnn_c.py
python scripts/download_datasets/omniobject3d.py
python scripts/download_datasets/modelnet40.py
python scripts/download_datasets/scanobjectnn.py
python scripts/download_datasets/objaverse_lvis.py
```

To verify a specific dataset after downloading:

```bash
python scripts/download_datasets/modelnet40_c.py --verify-only
python scripts/download_datasets/modelnet40.py --verify-only
```

## Verification

For file-tree datasets, verification includes:

- file count check
- file size check
- SHA-256 checksum check when HuggingFace LFS metadata is available

For `objaverse_lvis`, the script downloads `objaverse_lvis.zip`, verifies its size and SHA-256 checksum, checks the zip CRC, and then extracts it to `data/objaverse_lvis`.

Expected success message for file-tree datasets:

```text
校验通过：文件数量、文件大小、SHA-256 内容哈希均匹配。
<dataset-name> 下载并校验完成！
```

If verification fails, the script prints missing files, size-mismatched files, or checksum-mismatched files. Run the normal download command again to repair missing or incomplete files.

## Notes

- The scripts depend on `wget` for file downloading.
- The HuggingFace mirror directory tree API may return `HTTP 308 Permanent Redirect`. The shared utility handles this redirect automatically.
- Do not manually rename downloaded files. Dataset loader code expects the file names and directory layout provided by the remote repositories.
