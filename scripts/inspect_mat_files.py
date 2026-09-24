"""
Standalone diagnostic: inspect every .mat file under data/raw/figshare_mat/ and
report which ones are valid, invalid, or unreadable — with the reason for each.

This is the script to run FIRST if build_metadata() raises a validation error and
you want to see the full picture (not just the first 10 files in the error message).

Usage:
    python scripts/inspect_mat_files.py
    python scripts/inspect_mat_files.py --dir path/to/figshare_mat
"""

import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_utils import _get_cjdata_group  # noqa: E402


def inspect_directory(figshare_dir: Path) -> None:
    mat_files = sorted(figshare_dir.rglob("*.mat"))

    if not mat_files:
        print(f"No .mat files found under {figshare_dir} (searched recursively).")
        print("Check that the Figshare zip parts were extracted here — see data/README.md.")
        return

    print(f"Found {len(mat_files)} .mat file(s) under {figshare_dir}\n")

    n_ok, n_bad = 0, 0
    for mat_path in mat_files:
        rel_path = mat_path.relative_to(figshare_dir)

        if not h5py.is_hdf5(mat_path):
            print(f"  [BAD]  {rel_path}")
            print("         FORMAT: not HDF5/MATLAB v7.3 — h5py cannot read this file.")
            print(
                "         This usually means the file is an older MATLAB format, "
                "or the download/extraction is corrupted."
            )
            n_bad += 1
            continue

        try:
            with h5py.File(mat_path, "r") as f:
                cjdata = _get_cjdata_group(f, mat_path)
                expected_fields = {"label", "PID", "image", "tumorMask"}
                present_fields = set(cjdata.keys())
                missing = expected_fields - present_fields
                if missing:
                    print(f"  [BAD]  {rel_path}")
                    print(f"         HAS cjdata: True, but missing fields: {missing}")
                    n_bad += 1
                else:
                    n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [BAD]  {rel_path}")
            print(f"         {e}")
            n_bad += 1

    print(f"\nSummary: {n_ok} valid, {n_bad} invalid, out of {len(mat_files)} total.")
    if n_bad > 0:
        print(
            "\nFor invalid files: re-download and re-extract the 4 Figshare zip parts "
            "from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 "
            "into data/raw/figshare_mat/ (see data/README.md)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw" / "figshare_mat",
        help="Path to the figshare_mat directory (default: data/raw/figshare_mat)",
    )
    args = parser.parse_args()
    inspect_directory(args.dir)
