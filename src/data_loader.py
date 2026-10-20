"""Pemuatan citra Sentinel-1 (radar) dan Sentinel-2 (optik) - Sesi 6.

Modul ini menangani:
- Pembacaan raster GeoTIFF via rasterio (metadata, CRS, resolusi, kanal).
- Penanganan NoData dan nilai tidak valid.
- Ekstraksi fitur: backscatter radar (S1) dan fitur spektral + NDVI (S2).
- Pembuatan dataset demo sintetik (diberi label jelas) untuk perbandingan S1 vs S2.

PENTING: Sentinel-1 adalah radar (VV/VH), Sentinel-2 optik (kanal spektral).
Jangan menyamakan keduanya; fitur dan cara interpretasinya berbeda.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

from config.kelas import SENTINEL1_KANAL, SENTINEL2_KANAL


@dataclass
class InfoRaster:
    path: str
    lebar: int
    tinggi: int
    jumlah_kanal: int
    crs: str | None
    resolusi: tuple | None
    bounds: tuple | None
    nodata: float | None
    tipe_data: str
    deskripsi_kanal: list = field(default_factory=list)
    jenis_sensor: str = "tidak diketahui"

    def sebagai_dict(self) -> dict:
        return {
            "path": self.path,
            "lebar": self.lebar,
            "tinggi": self.tinggi,
            "jumlah_kanal": self.jumlah_kanal,
            "crs": self.crs,
            "resolusi": self.resolusi,
            "bounds": self.bounds,
            "nodata": self.nodata,
            "tipe_data": self.tipe_data,
            "deskripsi_kanal": self.deskripsi_kanal,
            "jenis_sensor": self.jenis_sensor,
        }


def _tebak_sensor(jumlah_kanal: int, deskripsi: list) -> str:
    gabung = " ".join(d.upper() for d in deskripsi)
    if any(k in gabung for k in ("VV", "VH", "SAR", "SIGMA", "AMPLITUDE")):
        return "Sentinel-1 (radar)"
    if any(k in gabung for k in ("B02", "B03", "B04", "B08", "NIR", "RED", "GREEN", "BLUE")):
        return "Sentinel-2 (optik)"
    if jumlah_kanal <= 2:
        return "Sentinel-1 (radar)"
    return "Sentinel-2 (optik)"


def baca_info_raster(path: str | Path) -> InfoRaster:
    if rasterio is None:
        raise ImportError("rasterio belum terpasang. Install: pip install rasterio")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster tidak ditemukan: {path}")
    with rasterio.open(path) as src:
        deskripsi = [d or f"Band-{i + 1}" for i, d in enumerate(src.descriptions)]
        return InfoRaster(
            path=str(path),
            lebar=src.width,
            tinggi=src.height,
            jumlah_kanal=src.count,
            crs=str(src.crs) if src.crs else None,
            resolusi=src.res,
            bounds=tuple(src.bounds),
            nodata=src.nodata,
            tipe_data=src.dtypes[0] if src.dtypes else "unknown",
            deskripsi_kanal=deskripsi,
            jenis_sensor=_tebak_sensor(src.count, deskripsi),
        )


def baca_raster(path: str | Path, max_piksel: int | None = 300_000) -> np.ndarray:
    """Membaca raster multi-kanal sebagai (kanal, tinggi, lebar). Menangani NoData."""
    if rasterio is None:
        raise ImportError("rasterio belum terpasang. Install: pip install rasterio")
    with rasterio.open(path) as src:
        step = 1
        if max_piksel is not None and src.width * src.height > max_piksel:
            step = int(np.ceil(np.sqrt(src.width * src.height / max_piksel)))
        data = src.read(
            out_shape=(src.count, int(src.height / step), int(src.width / step)),
            resampling=rasterio.enums.Resampling.nearest,
        )
        nodata = src.nodata
    arr = data.astype(np.float32)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def normalisasi_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalisasi 0-1 per kanal, mengabaikan NaN."""
    out = arr.copy()
    for c in range(out.shape[0]):
        band = out[c]
        lo, hi = np.nanmin(band), np.nanmax(band)
        if hi - lo < 1e-8:
            out[c] = 0.0
        else:
            out[c] = (band - lo) / (hi - lo)
    return out


def fitur_sentinel1(arr: np.ndarray, nama_kanal: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Fitur radar: backscatter VV/VH (dB) dan rasio VH/VV."""
    nama_kanal = nama_kanal or SENTINEL1_KANAL[: arr.shape[0]]
    fitur_list = [arr[c].ravel() for c in range(arr.shape[0])]
    nama_fitur = list(nama_kanal)
    if arr.shape[0] >= 2:
        vv, vh = arr[0], arr[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rasio = vh - vv  # dalam skala dB, selisih = rasio
        rasio[~np.isfinite(rasio)] = np.nan
        fitur_list.append(rasio.ravel())
        nama_fitur.append("VH_minus_VV")
    return np.vstack(fitur_list).T.astype(np.float32), nama_fitur


def fitur_sentinel2(arr: np.ndarray, nama_kanal: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Fitur optik: kanal spektral + NDVI (jika Red & NIR tersedia)."""
    nama_kanal = nama_kanal or SENTINEL2_KANAL[: arr.shape[0]]
    fitur_list = [arr[c].ravel() for c in range(arr.shape[0])]
    nama_fitur = list(nama_kanal)

    idx_red = next((i for i, n in enumerate(nama_kanal) if "RED" in n.upper() or n.upper() == "B04"), None)
    idx_nir = next((i for i, n in enumerate(nama_kanal) if "NIR" in n.upper() or n.upper() == "B08"), None)
    if idx_red is not None and idx_nir is not None:
        red, nir = arr[idx_red], arr[idx_nir]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
        ndvi[~np.isfinite(ndvi)] = np.nan
        fitur_list.append(np.clip(ndvi, -1, 1).ravel())
        nama_fitur.append("NDVI")
    return np.vstack(fitur_list).T.astype(np.float32), nama_fitur


def tangani_nan(fitur: np.ndarray, strategi: str = "median") -> np.ndarray:
    fitur = fitur.copy()
    if strategi == "nol":
        return np.nan_to_num(fitur, nan=0.0)
    median = np.nanmedian(fitur, axis=0)
    inds = np.where(np.isnan(fitur))
    fitur[inds] = np.take(median, inds[1])
    return np.nan_to_num(fitur, nan=0.0)


def buat_dataset_demo(
    n_sampel: int = 2000, jenis: str = "S2", seed: int = 42, tutupan_awal: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Dataset perubahan tutupan hutan sintetik untuk MODE DEMO.

    BUKAN citra Sentinel nyata. Kelas biner: 0 = bukan hutan, 1 = hutan.
    `jenis` = 'S1' (radar) atau 'S2' (optik). Jika `tutupan_awal` True, distribusi
    dibuat menyerupai tutupan hutan; jika False menyerupai perubahan.
    """
    rng = np.random.default_rng(seed)
    n_kelas = 2

    if jenis.upper() == "S1":
        nama_kanal = SENTINEL1_KANAL
        pusat = {
            0: {"VV": -8.5, "VH": -14.0},   # bukan hutan: backscatter lebih rendah
            1: {"VV": -7.0, "VH": -10.5},   # hutan: volume scattering tinggi (VH naik)
        }
        spread = {0: {"VV": 1.0, "VH": 1.2}, 1: {"VV": 1.0, "VH": 1.1}}
    else:
        nama_kanal = SENTINEL2_KANAL
        pusat = {
            0: {"B02_Blue": 0.12, "B03_Green": 0.15, "B04_Red": 0.18, "B08_NIR": 0.25},
            1: {"B02_Blue": 0.05, "B03_Green": 0.12, "B04_Red": 0.06, "B08_NIR": 0.42},
        }
        spread = {
            0: {"B02_Blue": 0.03, "B03_Green": 0.03, "B04_Red": 0.04, "B08_NIR": 0.05},
            1: {"B02_Blue": 0.02, "B03_Green": 0.03, "B04_Red": 0.02, "B08_NIR": 0.07},
        }

    X, y = [], []
    per_kelas = n_sampel // n_kelas
    for kelas in range(n_kelas):
        baris = [rng.normal(pusat[kelas][k], spread[kelas][k], per_kelas) for k in nama_kanal]
        X.append(np.vstack(baris).T)
        y.append(np.full(per_kelas, kelas))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y)
    if jenis.upper() != "S1":
        X = np.clip(X, 0, 1)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], list(nama_kanal)
