"""
Data loading and patient-level splitting utilities.

The patient-level split is the core methodological fix for this project: the widely
used Kaggle version of this dataset splits at the slice level, allowing near-identical
slices from the same patient to appear in both train and test sets. Every function here
that touches splitting is designed to make that mistake structurally hard to repeat.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}
NO_TUMOR_LABEL = "no_tumor"


def _decode_matlab_string(h5_dataset) -> str:
    """
    Decodes a MATLAB string stored in HDF5 (.mat v7.3) format.

    MATLAB v7.3 .mat files store strings as an array of character codes
    (one uint16 per character), not as a plain scalar or byte string —
    this is how the Cheng (2017) dataset's PID field is stored. A naive
    `.item()` call on this array fails since it isn't a size-1 array.
    """
    char_codes = np.array(h5_dataset).flatten()
    return "".join(chr(int(code)) for code in char_codes)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42


def _get_cjdata_group(h5_file: h5py.File, file_path: Path):
    """
    Returns the `cjdata` group from a Figshare MATLAB v7.3 (.mat) file.

    Cheng (2017)'s .mat files store `cjdata` at the root. This checks the root
    first, falls back to a one-level-deep search (in case an extraction tool
    nested it under a container group), and — critically — raises an error
    that names the actual file and shows what keys ARE present, rather than a
    bare KeyError. A bad/missing-group file is a genuine data problem (wrong
    download, corrupted extraction, wrong dataset entirely), not something to
    silently paper over — this only makes the failure identifiable, not hidden.
    """
    if "cjdata" in h5_file:
        return h5_file["cjdata"]

    for key in h5_file.keys():
        item = h5_file[key]
        if isinstance(item, h5py.Group) and "cjdata" in item:
            logger.warning(
                "%s: found 'cjdata' nested under '%s' instead of at the root — "
                "this is unusual for the Cheng (2017) dataset; verify this file's source.",
                file_path,
                key,
            )
            return item["cjdata"]

    raise ValueError(
        f"'{file_path}' does not contain a 'cjdata' group at the root or one "
        f"level deep. Top-level keys found: {list(h5_file.keys())}. This usually "
        f"means: (1) this isn't a Cheng (2017) Figshare .mat file, (2) the download "
        f"or extraction is incomplete/corrupted, or (3) the file is an older MATLAB "
        f"format (pre-v7.3) that h5py cannot read — check with `h5py.is_hdf5(path)`. "
        f"See data/README.md for the expected source and structure."
    )


def validate_figshare_directory(figshare_dir: Path) -> None:
    """
    Validates every .mat file under figshare_dir before build_metadata() commits
    to processing them — fails fast with a clear, per-file report rather than
    stopping partway through a large batch on an unhelpful error.
    """
    mat_files = sorted(figshare_dir.rglob("*.mat"))
    if not mat_files:
        raise ValueError(
            f"No .mat files found under {figshare_dir} (searched recursively). "
            f"Check that the Figshare zip parts were extracted here — see data/README.md."
        )

    bad_files: list[tuple[Path, str]] = []
    for mat_path in mat_files:
        try:
            if not h5py.is_hdf5(mat_path):
                bad_files.append((mat_path, "not an HDF5/MATLAB v7.3 file"))
                continue
            with h5py.File(mat_path, "r") as f:
                _get_cjdata_group(f, mat_path)
        except Exception as e:  # noqa: BLE001 — intentionally broad: report and continue
            bad_files.append((mat_path, str(e)))

    if bad_files:
        report = "\n".join(f"  - {path}: {reason}" for path, reason in bad_files[:10])
        more = f"\n  ... and {len(bad_files) - 10} more" if len(bad_files) > 10 else ""
        raise ValueError(
            f"{len(bad_files)} of {len(mat_files)} .mat file(s) under {figshare_dir} "
            f"failed validation:\n{report}{more}\n\n"
            f"See data/README.md for the expected download source and structure."
        )

    logger.info("Validated %d .mat files under %s — all OK.", len(mat_files), figshare_dir)


def load_figshare_mat(file_path: Path) -> dict:
    """
    Loads a single Cheng (2017) Figshare .mat file.

    Note: these files are saved in HDF5-based MATLAB v7.3 format, so `h5py` is used
    rather than `scipy.io.loadmat` (which only supports older .mat versions).

    Returns
    -------
    dict with keys: image (np.ndarray), label (str), patient_id (str), tumor_mask (np.ndarray)
    """
    with h5py.File(file_path, "r") as f:
        cjdata = _get_cjdata_group(f, file_path)
        label_code = int(np.array(cjdata["label"]).item())
        patient_id = _decode_matlab_string(cjdata["PID"])
        image = np.array(cjdata["image"])
        tumor_mask = np.array(cjdata["tumorMask"])

    return {
        "image": image,
        "label": LABEL_MAP[label_code],
        "patient_id": f"figshare_{patient_id}",
        "tumor_mask": tumor_mask,
        "source_dataset": "figshare",
    }


def build_metadata(
    figshare_dir: Path,
    br35h_no_tumor_dir: Path,
    output_csv: Path,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Scans both raw data sources and builds the unified metadata table that drives
    the rest of the pipeline. Does NOT load full image arrays into this table —
    only paths and labels — so this stays cheap to run and re-run.

    Scans recursively (rglob), so an extra nested folder from zip extraction
    (e.g. figshare_mat/brain_tumor_dataset/*.mat) won't silently produce zero rows.

    validate=True (default) runs validate_figshare_directory() first, so a bad
    download fails fast with a clear per-file report instead of partway through.
    """
    if validate:
        validate_figshare_directory(figshare_dir)

    records = []

    for mat_path in sorted(figshare_dir.rglob("*.mat")):
        with h5py.File(mat_path, "r") as f:
            cjdata = _get_cjdata_group(f, mat_path)
            label_code = int(np.array(cjdata["label"]).item())
            patient_id = _decode_matlab_string(cjdata["PID"])
        records.append(
            {
                "image_path": str(mat_path),
                "label": LABEL_MAP[label_code],
                "patient_id": f"figshare_{patient_id}",
                "source_dataset": "figshare",
            }
        )

    for img_path in sorted(br35h_no_tumor_dir.rglob("*.jpg")) + sorted(
        br35h_no_tumor_dir.rglob("*.png")
    ):
        records.append(
            {
                "image_path": str(img_path),
                "label": NO_TUMOR_LABEL,
                # Each no-tumor image is its own "patient" group — there's no real
                # patient linkage in Br35H, and giving each a unique ID means it
                # can never leak across splits (a group of size 1 can only ever
                # land entirely in one split).
                "patient_id": f"br35h_{uuid.uuid4().hex[:12]}",
                "source_dataset": "br35h",
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No images found. Check data/README.md for expected raw data locations.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote metadata for %d images to %s", len(df), output_csv)
    return df


def patient_level_split(metadata: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    """
    Assigns each row a 'split' value (train/val/test) such that all rows sharing a
    patient_id land in the same split. Returns the metadata frame with a new
    'split' column — does not load or move any image files.
    """
    if config is None:
        config = SplitConfig()

    groups = metadata["patient_id"].values
    labels = metadata["label"].values
    indices = np.arange(len(metadata))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    train_val_idx, test_idx = next(gss1.split(indices, labels, groups=groups))

    relative_val_size = config.val_size / (1 - config.test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=config.seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(train_val_idx, labels[train_val_idx], groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    split_col = np.array(["unassigned"] * len(metadata), dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    result = metadata.copy()
    result["split"] = split_col

    assert_no_patient_leakage(result)
    return result


def load_image_as_array(
    image_path: str, source_dataset: str, img_size: tuple[int, int]
) -> np.ndarray:
    """
    Loads a single image (either a Figshare .mat file or a Br35H .jpg/.png),
    resizes to img_size, and returns a float32 array in [0, 1] with shape
    (H, W, 1). Source-agnostic caller code (build_tf_dataset below) doesn't
    need to know which format it's reading.
    """
    if source_dataset == "figshare":
        with h5py.File(image_path, "r") as f:
            cjdata = _get_cjdata_group(f, Path(image_path))
            image = np.array(cjdata["image"], dtype=np.float32)
    else:
        from PIL import Image

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

    # Normalize to [0, 1] using this image's own range — MRI intensity scales
    # vary by scan/source, so per-image min-max is more robust than assuming
    # a fixed 0-255 range (which Figshare's raw pixel data does not follow).
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    image_resized = np.array(tf.image.resize(image[..., np.newaxis], img_size))
    return image_resized.astype("float32")


def build_tf_dataset(
    split_metadata: pd.DataFrame,
    split_name: str,
    class_names: list[str],
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset for one split ('train', 'val', or 'test') from the
    split metadata table. Uses tf.py_function to wrap the h5py/PIL loading logic
    above, since neither format is readable by pure TF ops directly.

    Yields (image, label_index) batches, image shape (H, W, 1), label as int.
    """
    rows = split_metadata[split_metadata["split"] == split_name].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows found for split='{split_name}'")

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    paths = rows["image_path"].values
    sources = rows["source_dataset"].values
    labels = rows["label"].map(label_to_idx).values.astype("int32")

    def _load(path_tensor, source_tensor, label_tensor):
        path = path_tensor.numpy().decode("utf-8")
        source = source_tensor.numpy().decode("utf-8")
        image = load_image_as_array(path, source, img_size)
        return image, label_tensor

    def _tf_load(path_tensor, source_tensor, label_tensor):
        image, label = tf.py_function(
            _load, [path_tensor, source_tensor, label_tensor], [tf.float32, tf.int32]
        )
        image.set_shape((*img_size, 1))
        label.set_shape(())
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, sources, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(rows), seed=seed)
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def assert_no_patient_leakage(metadata_with_split: pd.DataFrame) -> None:
    """
    Hard safety check: raises if any patient_id appears in more than one split.
    This is the automated version of the manual check that the leaky Kaggle
    dataset never had — call this in tests AND at the end of the split notebook.
    """
    grouped = metadata_with_split.groupby("patient_id")["split"].nunique()
    leaking_patients = grouped[grouped > 1]
    if not leaking_patients.empty:
        raise ValueError(
            f"Patient-level leakage detected for {len(leaking_patients)} patient(s): "
            f"{list(leaking_patients.index)[:10]}..."
        )
    logger.info("Leakage check passed: no patient appears in more than one split.")
