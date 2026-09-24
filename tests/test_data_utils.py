"""
Tests for src/data_utils.py.

The most important test in this file — and arguably in the whole repo — is
test_no_patient_leakage_across_splits. It's a direct, automated guard against
the exact bug that made the public Kaggle version of this dataset unreliable.

test_load_figshare_mat_decodes_pid_correctly guards a second real bug found
during development: the Cheng (2017) .mat files store PID as a MATLAB string
(an array of character codes in HDF5 format), not a scalar. A naive .item()
call crashes on real data — this test uses a synthetic .mat file built with
the same HDF5 structure to catch that regression without needing real data.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    SplitConfig,
    _decode_matlab_string,
    _get_cjdata_group,
    assert_no_patient_leakage,
    build_metadata,
    build_tf_dataset,
    load_figshare_mat,
    patient_level_split,
    validate_figshare_directory,
)


def _write_synthetic_mat(path: Path, label_code: int, patient_id: str, size: int = 32) -> None:
    """Builds a .mat file matching the real Cheng (2017) HDF5 structure —
    PID as char codes, not a scalar — so tests exercise the real format."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("cjdata")
        grp.create_dataset("label", data=np.array([[float(label_code)]]))
        pid_codes = np.array([[ord(c)] for c in patient_id], dtype=np.uint16)
        grp.create_dataset("PID", data=pid_codes)
        grp.create_dataset("image", data=np.random.rand(size, size))
        grp.create_dataset("tumorMask", data=np.random.randint(0, 2, (size, size)))


def make_fake_metadata(n_patients: int = 40, slices_per_patient: int = 5) -> pd.DataFrame:
    """Builds synthetic metadata mimicking multiple slices per patient, without
    needing real .mat files — keeps this test fast and independent of data availability."""
    records = []
    labels = ["glioma", "meningioma", "pituitary", "no_tumor"]
    for i in range(n_patients):
        patient_id = f"patient_{i}"
        label = labels[i % len(labels)]
        for s in range(slices_per_patient):
            records.append(
                {
                    "image_path": f"fake/{patient_id}_slice{s}.mat",
                    "label": label,
                    "patient_id": patient_id,
                    "source_dataset": "figshare",
                }
            )
    return pd.DataFrame.from_records(records)


def test_no_patient_leakage_across_splits():
    """The core regression test: no patient_id should ever span more than one split."""
    metadata = make_fake_metadata()
    result = patient_level_split(metadata, SplitConfig(test_size=0.2, val_size=0.1, seed=42))

    # Should not raise
    assert_no_patient_leakage(result)

    # Explicit cross-check in the test itself, independent of the function under test
    patient_to_splits = result.groupby("patient_id")["split"].nunique()
    assert (patient_to_splits == 1).all(), "Found a patient spanning multiple splits"


def test_split_covers_every_row():
    metadata = make_fake_metadata()
    result = patient_level_split(metadata)
    assert not (result["split"] == "unassigned").any()


def test_split_proportions_roughly_correct():
    metadata = make_fake_metadata(n_patients=100, slices_per_patient=1)
    result = patient_level_split(metadata, SplitConfig(test_size=0.2, val_size=0.1, seed=42))
    proportions = result["split"].value_counts(normalize=True)

    # Loose bounds — patient-level grouping means exact percentages aren't guaranteed
    assert 0.6 <= proportions["train"] <= 0.8
    assert 0.05 <= proportions["val"] <= 0.2
    assert 0.1 <= proportions["test"] <= 0.3


def test_decode_matlab_string_handles_char_code_array():
    """_decode_matlab_string must turn a per-character code array (the real
    on-disk format for MATLAB strings) back into a plain string, not crash
    on a naive .item() call the way the original buggy code did."""
    pid_codes = np.array([[ord(c)] for c in "5678"], dtype=np.uint16)
    assert _decode_matlab_string(pid_codes) == "5678"


def test_load_figshare_mat_decodes_pid_correctly(tmp_path):
    """Regression test for the PID-parsing bug found during development:
    PID is a char-code array, not a scalar — a naive .item() call crashes."""
    mat_path = tmp_path / "sample.mat"
    _write_synthetic_mat(mat_path, label_code=2, patient_id="1234")

    result = load_figshare_mat(mat_path)

    assert result["label"] == "glioma"
    assert result["patient_id"] == "figshare_1234"
    assert result["image"].shape == (32, 32)
    assert result["tumor_mask"].shape == (32, 32)


def test_get_cjdata_group_missing_raises_clear_error_with_filename(tmp_path):
    """Regression test for the vague-error problem: a .mat file with no
    cjdata group at all should raise an error naming the actual file and
    what keys WERE found, not a bare KeyError."""
    bad_path = tmp_path / "invalid.mat"
    with h5py.File(bad_path, "w") as f:
        f.create_group("wrong_group")

    with h5py.File(bad_path, "r") as f:
        with pytest.raises(ValueError, match=r"invalid\.mat.*cjdata"):
            _get_cjdata_group(f, bad_path)


def test_get_cjdata_group_one_level_nested_is_supported(tmp_path):
    """Some extraction/conversion tools may place cjdata one level below
    the root — the one-level fallback search should still find it."""
    nested_path = tmp_path / "nested.mat"
    with h5py.File(nested_path, "w") as f:
        container = f.create_group("container")
        cjdata = container.create_group("cjdata")
        cjdata.create_dataset("label", data=np.array([[2.0]]))

    with h5py.File(nested_path, "r") as f:
        result = _get_cjdata_group(f, nested_path)
        assert "label" in result


def test_load_figshare_mat_raises_clear_error_on_non_cheng_file(tmp_path):
    """A structurally unrelated .mat file (valid HDF5, but not this dataset's
    format) should fail with a message pointing at the actual cause, not a
    generic KeyError several layers removed from what went wrong."""
    unrelated_path = tmp_path / "unrelated.mat"
    with h5py.File(unrelated_path, "w") as f:
        f.create_dataset("some_other_field", data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="cjdata"):
        load_figshare_mat(unrelated_path)


def test_validate_figshare_directory_passes_on_good_files(tmp_path):
    figshare_dir = tmp_path / "figshare"
    figshare_dir.mkdir()
    _write_synthetic_mat(figshare_dir / "p1.mat", label_code=1, patient_id="0001")
    _write_synthetic_mat(figshare_dir / "p2.mat", label_code=2, patient_id="0002")

    validate_figshare_directory(figshare_dir)  # should not raise


def test_validate_figshare_directory_reports_bad_file_by_name(tmp_path):
    figshare_dir = tmp_path / "figshare"
    figshare_dir.mkdir()
    _write_synthetic_mat(figshare_dir / "good.mat", label_code=1, patient_id="0001")
    with h5py.File(figshare_dir / "bad.mat", "w") as f:
        f.create_group("wrong_group")

    with pytest.raises(ValueError, match="bad.mat"):
        validate_figshare_directory(figshare_dir)


def test_validate_figshare_directory_raises_on_empty_dir(tmp_path):
    figshare_dir = tmp_path / "empty_figshare"
    figshare_dir.mkdir()

    with pytest.raises(ValueError, match="No .mat files found"):
        validate_figshare_directory(figshare_dir)


def test_build_metadata_scans_nested_directories(tmp_path):
    """Regression test: if zip extraction produced an extra nested folder
    (e.g. figshare_mat/brain_tumor_dataset/*.mat), build_metadata should
    still find the files via rglob, not silently return zero rows."""
    from PIL import Image

    figshare_dir = tmp_path / "figshare_mat"
    nested_dir = figshare_dir / "brain_tumor_dataset" / "part1"
    nested_dir.mkdir(parents=True)
    _write_synthetic_mat(nested_dir / "p1.mat", label_code=1, patient_id="0001")

    br35h_dir = tmp_path / "br35h_no_tumor"
    br35h_nested = br35h_dir / "no"
    br35h_nested.mkdir(parents=True)
    Image.new("L", (32, 32)).save(br35h_nested / "healthy1.jpg")

    metadata = build_metadata(figshare_dir, br35h_dir, tmp_path / "metadata.csv")

    assert len(metadata) == 2
    assert set(metadata["label"]) == {"meningioma", "no_tumor"}


def test_build_metadata_end_to_end(tmp_path):
    """Builds a tiny synthetic dataset (2 figshare patients with 2 slices
    each, 2 br35h no-tumor images) and verifies the full metadata pipeline,
    including that br35h images each get their own unique, non-colliding
    patient_id (so they can never leak across splits)."""
    figshare_dir = tmp_path / "figshare_mat"
    br35h_dir = tmp_path / "br35h_no_tumor"
    figshare_dir.mkdir()
    br35h_dir.mkdir()

    _write_synthetic_mat(figshare_dir / "p1_s1.mat", label_code=1, patient_id="0001")
    _write_synthetic_mat(figshare_dir / "p1_s2.mat", label_code=1, patient_id="0001")
    _write_synthetic_mat(figshare_dir / "p2_s1.mat", label_code=3, patient_id="0002")

    from PIL import Image

    Image.new("L", (32, 32)).save(br35h_dir / "healthy1.jpg")
    Image.new("L", (32, 32)).save(br35h_dir / "healthy2.jpg")

    output_csv = tmp_path / "metadata.csv"
    df = build_metadata(figshare_dir, br35h_dir, output_csv)

    assert len(df) == 5
    assert output_csv.exists()
    assert set(df["label"]) == {"meningioma", "pituitary", "no_tumor"}

    # The two figshare slices from patient 0001 must share one patient_id
    p1_rows = df[df["patient_id"] == "figshare_0001"]
    assert len(p1_rows) == 2

    # The two br35h images must NOT share a patient_id with each other
    br35h_ids = df[df["source_dataset"] == "br35h"]["patient_id"]
    assert br35h_ids.nunique() == 2


def test_build_tf_dataset_end_to_end(tmp_path):
    """Full pipeline: synthetic figshare + br35h files -> metadata -> patient
    split -> tf.data.Dataset, checking image shape, dtype, and normalization."""
    from PIL import Image

    figshare_dir = tmp_path / "figshare_mat"
    br35h_dir = tmp_path / "br35h_no_tumor"
    figshare_dir.mkdir()
    br35h_dir.mkdir()

    for i in range(6):
        _write_synthetic_mat(
            figshare_dir / f"p{i}.mat", label_code=(i % 3) + 1, patient_id=f"{100 + i}", size=64
        )
    for i in range(4):
        Image.new("L", (64, 64), color=int(50 * i)).save(br35h_dir / f"healthy{i}.jpg")

    metadata = build_metadata(figshare_dir, br35h_dir, tmp_path / "metadata.csv")
    split_metadata = patient_level_split(
        metadata, SplitConfig(test_size=0.3, val_size=0.15, seed=1)
    )

    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    train_split = split_metadata[split_metadata["split"] == "train"]
    if train_split.empty:
        pytest.skip("Synthetic dataset too small to produce a non-empty train split")

    ds = build_tf_dataset(
        split_metadata, "train", class_names, img_size=(64, 64), batch_size=2, shuffle=False
    )
    images, labels = next(iter(ds))

    assert images.shape[1:] == (64, 64, 1)
    assert images.dtype.name == "float32"
    assert float(images.numpy().min()) >= 0.0
    assert float(images.numpy().max()) <= 1.0
    assert labels.dtype.name == "int32"


def test_build_tf_dataset_raises_on_empty_split(tmp_path):
    from PIL import Image

    figshare_dir = tmp_path / "figshare_mat"
    br35h_dir = tmp_path / "br35h_no_tumor"
    figshare_dir.mkdir()
    br35h_dir.mkdir()
    Image.new("L", (32, 32)).save(br35h_dir / "healthy.jpg")

    metadata = build_metadata(figshare_dir, br35h_dir, tmp_path / "metadata.csv", validate=False)
    metadata["split"] = "train"  # force everything into train, nothing in test

    with pytest.raises(ValueError, match="No rows found"):
        build_tf_dataset(metadata, "test", ["no_tumor"], img_size=(32, 32))


def test_assert_no_patient_leakage_raises_on_bad_input():
    """Sanity check that the assertion function actually catches a broken split,
    not just that it passes on good input."""
    bad_metadata = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "split": ["train", "test", "train"],  # p1 leaks across train/test
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_patient_leakage(bad_metadata)
