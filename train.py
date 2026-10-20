"""Entry point ForestWatch AI (Sesi 6).

Alur:
1. Siapkan data fitur S1 (radar) dan S2 (optik) - data nyata atau demo sintetik.
2. Latih & evaluasi Random Forest terpisah untuk masing-masing sensor.
3. Bandingkan performa S1 vs S2.
4. Simpan model, laporan, dan (opsional) peta perubahan.

Jalankan:
    python train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.kelas import NAMA_KELAS, NAMA_PERUBAHAN  # noqa: E402
from src.data_loader import buat_dataset_demo, fitur_sentinel1, fitur_sentinel2, tangani_nan  # noqa: E402
from src.model_ml import bandingkan_s1_s2, peta_perubahan, ringkasan_perubahan  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "outputs" / "results"


def _data_demo() -> dict:
    X1, y1, n1 = buat_dataset_demo(n_sampel=2000, jenis="S1")
    X2, y2, n2 = buat_dataset_demo(n_sampel=2000, jenis="S2")
    return {
        "S1": (tangani_nan(X1, "median"), y1, n1),
        "S2": (tangani_nan(X2, "median"), y2, n2),
    }


def main() -> int:
    print("=" * 70)
    print("ForestWatch AI - Perbandingan Sentinel-1 vs Sentinel-2 (Sesi 6)")
    print("=" * 70)

    print("[1/4] Menyiapkan data (mode demo sintetik)...")
    dataset = _data_demo()
    print(f"      S1 (radar): {dataset['S1'][0].shape}, fitur: {dataset['S1'][2]}")
    print(f"      S2 (optik): {dataset['S2'][0].shape}, fitur: {dataset['S2'][2]}")
    print("      CATATAN: data demo BUKAN citra Sentinel nyata.")

    print("[2/4] Melatih Random Forest per sensor...")
    hasil = bandingkan_s1_s2(dataset, n_estimators=200)

    print("[3/4] Perbandingan performa (F1 macro):")
    for sensor, r in hasil["per_sensor"].items():
        m = r["metrik"]
        print(
            f"      - {sensor}: accuracy={m['accuracy']:.4f} "
            f"precision={m['precision_macro']:.4f} recall={m['recall_macro']:.4f} "
            f"f1={m['f1_macro']:.4f}"
        )
    print(f"      Sensor terbaik (F1): {hasil['sensor_terbaik']}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for sensor, r in hasil["per_sensor"].items():
        joblib.dump(r["model"], str(MODELS_DIR / f"model_{sensor.lower()}.joblib"))

    # Contoh peta perubahan dari dua tutupan sintetik
    rng = np.random.default_rng(0)
    tutupan_awal = rng.integers(0, 2, (40, 40))
    tutupan_akhir = tutupan_awal.copy()
    hilang = rng.random((40, 40)) < 0.12
    tutupan_akhir[hilang & (tutupan_awal == 1)] = 0
    perub = peta_perubahan(tutupan_awal, tutupan_akhir)
    ringkas = ringkasan_perubahan(perub)

    laporan = {
        "mode": "demo_sintetik",
        "sensor_terbaik": hasil["sensor_terbaik"],
        "per_sensor": {
            s: {
                "nama_kanal": r["nama_kanal"],
                "n_fitur": r["metrik"]["n_fitur"],
                "n_train": r["metrik"]["n_train"],
                "n_test": r["metrik"]["n_test"],
                "metrik": r["metrik"],
            }
            for s, r in hasil["per_sensor"].items()
        },
        "contoh_peta_perubahan": ringkas,
        "kelas": {str(k): v for k, v in NAMA_KELAS.items()},
        "kelas_perubahan": {str(k): v for k, v in NAMA_PERUBAHAN.items()},
    }
    with open(RESULTS_DIR / "laporan_perbandingan.json", "w", encoding="utf-8") as fh:
        json.dump(laporan, fh, indent=2, ensure_ascii=False)

    print("[4/4] Tersimpan:")
    print(f"      - model : {MODELS_DIR}")
    print(f"      - laporan: {RESULTS_DIR / 'laporan_perbandingan.json'}")
    print("\nJalankan aplikasi: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
