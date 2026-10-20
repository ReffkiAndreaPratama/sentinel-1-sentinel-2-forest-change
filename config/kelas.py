"""Konfigurasi ForestWatch AI (Sesi 6)."""

from __future__ import annotations

KELAS = {
    0: {"nama": "Bukan Hutan", "warna": "#c9a227"},
    1: {"nama": "Hutan", "warna": "#2e7d32"},
}

KELAS_PERUBAHAN = {
    0: {"nama": "Stabil", "warna": "#9e9e9e"},
    1: {"nama": "Kehilangan Hutan (dari hutan)", "warna": "#b23b3b"},
    2: {"nama": "Bertambah Hutan (bukan -> hutan)", "warna": "#2e7d32"},
}

NAMA_KELAS = {k: v["nama"] for k, v in KELAS.items()}
WARNA_KELAS = {k: v["warna"] for k, v in KELAS.items()}
NAMA_PERUBAHAN = {k: v["nama"] for k, v in KELAS_PERUBAHAN.items()}
WARNA_PERUBAHAN = {k: v["warna"] for k, v in KELAS_PERUBAHAN.items()}

RANDOM_STATE = 42
TEST_SIZE = 0.3

SENTINEL1_KANAL = ["VV", "VH"]
SENTINEL2_KANAL = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]
