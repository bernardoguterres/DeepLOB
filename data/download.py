"""Utilities for obtaining and verifying the FI-2010 LOB dataset.

The FI-2010 dataset (Ntakaris et al., 2018) contains 10 days of high-frequency
limit order book data for 5 Finnish stocks traded on NASDAQ OMX Helsinki.

Usage::

    python -m data.download          # print instructions + verify data/raw/
    python -m data.download --verify # verify only
"""

import zipfile
from pathlib import Path


def print_instructions() -> None:
    """Print step-by-step instructions for obtaining the FI-2010 dataset."""
    print("""
FI-2010 Dataset — Download Instructions
========================================

The FI-2010 dataset is publicly available via the Finnish Fairdata service.

Step 1 — Open the dataset landing page:
    https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

Step 2 — Click "Go to original source" (Zenodo or direct link).
    Register / log in if prompted.

Step 3 — Download the zip archive.
    It is typically named:  BenchmarkDatasets.zip

Step 4 — Extract the .npy files into data/raw/:

    python -c "
    from data.download import extract_zip
    extract_zip('/path/to/BenchmarkDatasets.zip')
    "

    Or from the command line:
    python -m data.download --zip /path/to/BenchmarkDatasets.zip

Step 5 — Verify the extraction:

    python -m data.download --verify

Expected result: exactly 10 .npy files, each with shape[-1] == 144.

File layout after extraction:
    data/raw/
        ├── Train_Dst_NoAuction_ZScore_CF_7.npy
        ├── Train_Dst_NoAuction_ZScore_CF_8.npy
        ├── ... (10 files total, one per trading day)
        └── Train_Dst_NoAuction_ZScore_CF_16.npy
""")


def extract_zip(zip_path: str, target_dir: str = "data/raw/") -> None:
    """Extract .npy files from a FI-2010 zip archive into *target_dir*.

    Only ``.npy`` files are extracted; any directory structure inside the
    archive is flattened so all files land directly in *target_dir*.

    Args:
        zip_path: Absolute or relative path to the downloaded ``.zip`` archive.
        target_dir: Destination directory for extracted ``.npy`` files.
            Created automatically if it does not exist.

    Raises:
        FileNotFoundError: If *zip_path* does not exist.
    """
    zip_path_obj = Path(zip_path)
    if not zip_path_obj.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(zip_path_obj, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                continue
            # Flatten any nested paths inside the archive
            filename = Path(member).name
            dest = target / filename
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            print(f"  extracted → {dest}")

    print(f"\nDone. {extracted} .npy file(s) written to {target_dir}")


def verify_data(data_dir: str = "data/raw/") -> bool:
    """Verify that *data_dir* contains valid FI-2010 files.

    Checks performed:

    1. *data_dir* exists.
    2. Exactly 10 ``.npy`` files are present.
    3. Each file loads without error and has ``shape[-1] == 144``.

    Prints ``PASS`` / ``FAIL`` for every check and a summary line.

    Args:
        data_dir: Directory to inspect (default ``"data/raw/"``).

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    import numpy as np  # local import: keeps module lightweight when not verifying

    path = Path(data_dir)
    all_pass = True

    # ── Check 1: directory exists ────────────────────────────────────────────
    if not path.exists():
        print(f"FAIL  directory does not exist: {data_dir}")
        return False
    print(f"PASS  directory exists: {data_dir}")

    # ── Check 2: exactly 10 .npy files ──────────────────────────────────────
    npy_files = sorted(path.glob("*.npy"))
    n_files = len(npy_files)
    if n_files == 10:
        print("PASS  exactly 10 .npy files found")
    else:
        print(f"FAIL  expected 10 .npy files, found {n_files}")
        all_pass = False

    # ── Check 3: each file has shape[-1] == 144 ──────────────────────────────
    for fp in npy_files:
        try:
            arr = np.load(fp, mmap_mode="r")
            if arr.shape[-1] == 144:
                print(f"PASS  {fp.name:<45}  shape={arr.shape}")
            else:
                print(f"FAIL  {fp.name:<45}  shape={arr.shape}" f"  (expected shape[-1] == 144)")
                all_pass = False
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {fp.name:<45}  could not load: {exc}")
            all_pass = False

    # ── Summary ──────────────────────────────────────────────────────────────
    result = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n{result}")
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FI-2010 dataset utilities")
    parser.add_argument("--zip", metavar="PATH", help="Extract .npy files from this zip archive")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data/raw/ (run automatically when no other flag given)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/",
        metavar="DIR",
        help="Data directory to verify (default: data/raw/)",
    )
    args = parser.parse_args()

    if args.zip:
        extract_zip(args.zip, args.data_dir)

    if args.verify or not args.zip:
        print_instructions()
        verify_data(args.data_dir)
