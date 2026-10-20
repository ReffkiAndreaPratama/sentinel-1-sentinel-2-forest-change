"""Pengujian alur teknis ForestWatch AI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    buat_dataset_demo,
    fitur_sentinel1,
    fitur_sentinel2,
    normalisasi_minmax,
    tangani_nan,
)
from src.model_ml import (  # noqa: E402
    bandingkan_s1_s2,
    hitung_metrik,
    latih_dan_evaluasi,
    peta_perubahan,
    ringkasan_perubahan,
)


def test_dataset_s1_shapes() -> None:
    X, y, nama = buat_dataset_demo(n_sampel=400, jenis="S1")
    assert X.shape[1] == 2
    assert nama == ["VV", "VH"]
    assert set(np.unique(y)) == {0, 1}


def test_dataset_s2_shapes() -> None:
    X, y, nama = buat_dataset_demo(n_sampel=400, jenis="S2")
    assert X.shape[1] == 4
    assert "B08_NIR" in nama


def test_fitur_sentinel1() -> None:
    arr = np.random.randn(2, 8, 8).astype(np.float32)
    fitur, nama = fitur_sentinel1(arr, ["VV", "VH"])
    assert fitur.shape == (64, 3)
    assert "VH_minus_VV" in nama


def test_fitur_sentinel2_ndvi() -> None:
    arr = np.random.rand(4, 8, 8).astype(np.float32)
    fitur, nama = fitur_sentinel2(arr, ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"])
    assert fitur.shape == (64, 5)
    assert "NDVI" in nama


def test_normalisasi_minmax() -> None:
    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    norm = normalisasi_minmax(arr)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_tangani_nan() -> None:
    fitur = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 2.0]])
    hasil = tangani_nan(fitur, "median")
    assert not np.isnan(hasil).any()


def test_hitung_metrik() -> None:
    metrik = hitung_metrik([0, 1, 1, 0], [0, 1, 0, 0])
    assert 0.0 <= metrik["accuracy"] <= 1.0
    assert "confusion_matrix" in metrik


def test_peta_perubahan() -> None:
    awal = np.array([[1, 1], [0, 0]])
    akhir = np.array([[0, 1], [1, 0]])
    perub = peta_perubahan(awal, akhir)
    assert perub[0, 0] == 1  # hilang
    assert perub[1, 0] == 2  # tambah
    assert perub[0, 1] == 0  # stabil
    ringkas = ringkasan_perubahan(perub)
    assert ringkas["kehilangan_hutan"] == 1
    assert ringkas["pertambahan_hutan"] == 1


def test_bandingkan_s1_s2() -> None:
    X1, y1, n1 = buat_dataset_demo(n_sampel=400, jenis="S1")
    X2, y2, n2 = buat_dataset_demo(n_sampel=400, jenis="S2")
    dataset = {
        "S1": (tangani_nan(X1), y1, n1),
        "S2": (tangani_nan(X2), y2, n2),
    }
    hasil = bandingkan_s1_s2(dataset, n_estimators=30)
    assert "S1" in hasil["per_sensor"]
    assert "S2" in hasil["per_sensor"]
    assert hasil["sensor_terbaik"] in ("S1", "S2")
    assert hasil["per_sensor"]["S2"]["metrik"]["accuracy"] > 0.6


def test_latih_end_to_end() -> None:
    X, y, _ = buat_dataset_demo(n_sampel=300, jenis="S2")
    hasil = latih_dan_evaluasi(X, y, n_estimators=20)
    assert hasil["metrik"]["n_fitur"] == 4
