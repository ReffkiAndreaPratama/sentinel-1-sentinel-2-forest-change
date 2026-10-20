"""Klasifikasi dan perbandingan Sentinel-1 (radar) vs Sentinel-2 (optik) - Sesi 6.

- Model: Random Forest untuk klasifikasi tutupan hutan (biner).
- Perbandingan: melatih & mengevaluasi terpisah pada fitur S1 dan S2.
- Analisis perubahan tutupan hutan dari peta tutupan dua waktu.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.kelas import RANDOM_STATE, TEST_SIZE


def buat_pipeline(n_estimators: int = 200, max_depth: int | None = 16) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def bagi_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def hitung_metrik(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [int(x) for x in labels],
    }


def latih_dan_evaluasi(X, y, n_estimators: int = 200, max_depth: int | None = 16) -> dict:
    X_train, X_test, y_train, y_test = bagi_data(X, y)
    model = buat_pipeline(n_estimators, max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrik = hitung_metrik(y_test, y_pred)
    metrik["n_train"] = int(len(y_train))
    metrik["n_test"] = int(len(y_test))
    metrik["n_fitur"] = int(X.shape[1])
    return {"model": model, "metrik": metrik}


def bandingkan_s1_s2(dataset: dict, n_estimators: int = 200) -> dict:
    """Melatih & mengevaluasi model terpisah untuk S1 dan S2.

    `dataset` = {"S1": (X1, y1, nama1), "S2": (X2, y2, nama2)}.
    Mengembalikan hasil per sensor + sensor terbaik berdasarkan F1.
    """
    hasil = {}
    for sensor, (X, y, nama) in dataset.items():
        Xn = np.nan_to_num(X, nan=0.0)
        r = latih_dan_evaluasi(Xn, y, n_estimators=n_estimators)
        r["nama_kanal"] = list(nama)
        hasil[sensor] = r

    skor = {s: hasil[s]["metrik"]["f1_macro"] for s in hasil}
    terbaik = max(skor, key=skor.get) if skor else None
    return {"per_sensor": hasil, "skor_f1": skor, "sensor_terbaik": terbaik}


def peta_perubahan(tutupan_awal: np.ndarray, tutupan_akhir: np.ndarray) -> np.ndarray:
    """Menghitung peta perubahan dari dua peta tutupan (0/1).

    0 = stabil, 1 = kehilangan hutan (1->0), 2 = bertambah hutan (0->1).
    """
    awal = tutupan_awal.astype(int)
    akhir = tutupan_akhir.astype(int)
    perubahan = np.zeros_like(awal, dtype=int)
    perubahan[(awal == 1) & (akhir == 0)] = 1
    perubahan[(awal == 0) & (akhir == 1)] = 2
    return perubahan


def ringkasan_perubahan(perubahan: np.ndarray) -> dict:
    """Menghitung luas (piksel) setiap kelas perubahan."""
    total = perubahan.size
    stabil = int((perubahan == 0).sum())
    hilang = int((perubahan == 1).sum())
    tambah = int((perubahan == 2).sum())
    return {
        "total_piksel": int(total),
        "stabil": stabil,
        "kehilangan_hutan": hilang,
        "pertambahan_hutan": tambah,
        "persen_hilang": float(hilang / total * 100) if total else 0.0,
        "persen_tambah": float(tambah / total * 100) if total else 0.0,
    }
