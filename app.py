"""ForestWatch AI - Antarmuka Streamlit (Sesi 6).

Fitur:
- Muat citra Sentinel-1 (radar) dan/atau Sentinel-2 (optik), cek metadata.
- Ekstraksi fitur per sensor.
- Latih & bandingkan Random Forest S1 vs S2.
- Analisis perubahan tutupan hutan dari dua peta tutupan.
- Ekspor hasil.

Jalankan:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.kelas import NAMA_KELAS, NAMA_PERUBAHAN, WARNA_PERUBAHAN  # noqa: E402
from src.data_loader import (  # noqa: E402
    baca_info_raster,
    baca_raster,
    buat_dataset_demo,
    fitur_sentinel1,
    fitur_sentinel2,
    normalisasi_minmax,
    tangani_nan,
)
from src.model_ml import (  # noqa: E402
    bandingkan_s1_s2,
    peta_perubahan,
    ringkasan_perubahan,
)

st.set_page_config(page_title="ForestWatch AI", page_icon="🛰️", layout="wide")


def _chart_peta_label(peta: np.ndarray, nama_kelas: dict, judul: str) -> None:
    """Menampilkan peta label kelas sebagai citra berwarna."""
    h, w = peta.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    warna = {"Bukan Hutan": (201, 162, 39), "Hutan": (46, 125, 50)}
    for k, nama in nama_kelas.items():
        if nama in warna:
            rgb[peta == k] = warna[nama]
    st.image(rgb, caption=judul, use_container_width=True)


def render_demo() -> None:
    st.subheader("Mode Demo & Perbandingan Sensor")
    st.warning(
        "Data demo sintetik BUKAN citra Sentinel nyata. Hanya untuk memvalidasi pipeline "
        "perbandingan radar vs optik. Untuk hasil penelitian, unggah GeoTIFF nyata + label."
    )
    if st.button("Latih & Bandingkan S1 vs S2 (Demo)", type="primary"):
        with st.spinner("Melatih dua model (S1 & S2)..."):
            X1, y1, n1 = buat_dataset_demo(n_sampel=1200, jenis="S1")
            X2, y2, n2 = buat_dataset_demo(n_sampel=1200, jenis="S2")
            dataset = {
                "S1": (tangani_nan(X1, "median"), y1, n1),
                "S2": (tangani_nan(X2, "median"), y2, n2),
            }
            hasil = bandingkan_s1_s2(dataset, n_estimators=150)

        rows = []
        for sensor, r in hasil["per_sensor"].items():
            m = r["metrik"]
            rows.append(
                {
                    "Sensor": sensor,
                    "Fitur": ", ".join(r["nama_kanal"]),
                    "Accuracy": round(m["accuracy"], 4),
                    "Precision": round(m["precision_macro"], 4),
                    "Recall": round(m["recall_macro"], 4),
                    "F1 (macro)": round(m["f1_macro"], 4),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        fig = go.Figure()
        for metrik in ["Accuracy", "Precision", "Recall", "F1 (macro)"]:
            fig.add_trace(go.Bar(name=metrik, x=df["Sensor"], y=df[metrik]))
        fig.update_layout(barmode="group", title="Perbandingan S1 (radar) vs S2 (optik)")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Sensor terbaik berdasarkan F1: **{hasil['sensor_terbaik']}**")
        st.download_button(
            "Unduh Tabel Perbandingan (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="perbandingan_s1_s2.csv",
            mime="text/csv",
        )

        for sensor, r in hasil["per_sensor"].items():
            mat = np.array(r["metrik"]["confusion_matrix"])
            labels = [NAMA_KELAS.get(int(l), str(l)) for l in r["metrik"]["labels"]]
            fig = px.imshow(
                mat, text_auto=True, x=labels, y=labels,
                color_continuous_scale="Greens",
                title=f"Confusion Matrix - {sensor}",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_upload() -> None:
    st.subheader("Analisis Citra Sentinel (GeoTIFF)")
    sensor = st.radio("Jenis sensor", ["Sentinel-2 (optik)", "Sentinel-1 (radar)"], horizontal=True)
    upload = st.file_uploader("Unggah GeoTIFF", type=["tif", "tiff"])
    if not upload:
        st.info(
            "Unggah GeoTIFF Sentinel. Deskripsi kanal membantu sistem mengenali sensor "
            "(mis. B02/B03/B04/B08 untuk S2, VV/VH untuk S1)."
        )
        return

    tmp = ROOT / "outputs" / "_upload.tif"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(upload.getbuffer())
    info = baca_info_raster(tmp)
    st.json(info.sebagai_dict())

    try:
        arr = baca_raster(tmp)
    except Exception as exc:  # pragma: no cover
        st.error(f"Gagal membaca raster: {exc}")
        return

    st.markdown("**Pratinjau kanal**")
    n = min(arr.shape[0], 4)
    cols = st.columns(n)
    norm = normalisasi_minmax(arr)
    for i in range(n):
        cols[i].image(norm[i], caption=info.deskripsi_kanal[i], clamp=True, use_container_width=True)

    if sensor.startswith("Sentinel-1"):
        fitur, nama = fitur_sentinel1(arr, info.deskripsi_kanal)
    else:
        fitur, nama = fitur_sentinel2(arr, info.deskripsi_kanal)
    fitur = tangani_nan(fitur, "median")
    st.caption(f"Fitur diekstraksi: {nama} -> {fitur.shape}")

    model_path = ROOT / "models" / ("model_s1.joblib" if sensor.startswith("Sentinel-1") else "model_s2.joblib")
    if not model_path.exists():
        st.warning("Model belum tersedia. Latih dulu pada tab Demo atau `python train.py`.")
        return

    if st.button("Klasifikasi Tutupan Hutan", type="primary"):
        import joblib

        model = joblib.load(model_path)
        if fitur.shape[1] != model.n_features_in_:
            st.error(
                f"Jumlah fitur citra ({fitur.shape[1]}) tidak cocok dengan model "
                f"({model.n_features_in_}). Model demo dilatih pada kanal baku."
            )
            return
        pred = model.predict(fitur).reshape(arr.shape[1], arr.shape[2])
        _chart_peta_label(pred, NAMA_KELAS, "Peta tutupan hutan (diprediksi)")
        dist = pd.Series(pred.ravel()).value_counts().sort_index()
        df_dist = pd.DataFrame(
            {"kelas": [NAMA_KELAS.get(int(k), str(k)) for k in dist.index], "jumlah_piksel": dist.values}
        )
        fig = px.bar(df_dist, x="kelas", y="jumlah_piksel", color="kelas",
                     title="Distribusi kelas tutupan hutan")
        st.plotly_chart(fig, use_container_width=True)

        buf = io.BytesIO()
        np.save(buf, pred)
        st.download_button("Unduh Hasil Klasifikasi (.npy)", data=buf.getvalue(),
                           file_name="tutupan_hutan.npy", mime="application/octet-stream")


def render_perubahan() -> None:
    st.subheader("Analisis Perubahan Tutupan Hutan")
    st.caption(
        "Bandingkan dua peta tutupan (0 = bukan hutan, 1 = hutan) untuk mendeteksi "
        "kehilangan/pertambahan hutan. Perubahan tidak otomatis berarti deforestasi akibat "
        "aktivitas manusia."
    )
    metode = st.radio("Metode input", ["Contoh sintetik", "Unggah dua berkas .npy"], horizontal=True)
    awal = akhir = None
    if metode == "Contoh sintetik":
        rng = np.random.default_rng(0)
        awal = rng.integers(0, 2, (60, 60))
        akhir = awal.copy()
        akhir[(rng.random((60, 60)) < 0.15) & (awal == 1)] = 0
        akhir[(rng.random((60, 60)) < 0.05) & (awal == 0)] = 1
    else:
        f1 = st.file_uploader("Peta tutupan awal (.npy)", type=["npy"], key="awal")
        f2 = st.file_uploader("Peta tutupan akhir (.npy)", type=["npy"], key="akhir")
        if f1 and f2:
            awal = np.load(io.BytesIO(f1.getbuffer()))
            akhir = np.load(io.BytesIO(f2.getbuffer()))

    if awal is None or akhir is None:
        st.info("Pilih metode input untuk memulai analisis perubahan.")
        return
    if awal.shape != akhir.shape:
        st.error("Dimensi kedua peta harus sama.")
        return

    perub = peta_perubahan(awal, akhir)
    ringkas = ringkasan_perubahan(perub)
    c1, c2, c3 = st.columns(3)
    c1.metric("Kehilangan Hutan", f"{ringkas['kehilangan_hutan']} piksel ({ringkas['persen_hilang']:.2f}%)")
    c2.metric("Pertambahan Hutan", f"{ringkas['pertambahan_hutan']} piksel ({ringkas['persen_tambah']:.2f}%)")
    c3.metric("Stabil", f"{ringkas['stabil']} piksel")

    h, w = perub.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, nama in NAMA_PERUBAHAN.items():
        warna = tuple(int(WARNA_PERUBAHAN[k][i:i + 2], 16) for i in (1, 3, 5))
        rgb[perub == k] = warna
    st.image(rgb, caption="Peta perubahan (merah=kehilangan, hijau=pertambahan, abu=stabil)",
             use_container_width=True)

    df = pd.DataFrame(
        {"kelas": [NAMA_PERUBAHAN[k] for k in sorted(NAMA_PERUBAHAN)],
         "piksel": [ringkas["stabil"], ringkas["kehilangan_hutan"], ringkas["pertambahan_hutan"]]}
    )
    fig = px.pie(df, names="kelas", values="piksel", title="Proporsi Kelas Perubahan")
    st.plotly_chart(fig, use_container_width=True)


def render_laporan() -> None:
    path = ROOT / "outputs" / "results" / "laporan_perbandingan.json"
    if not path.exists():
        return
    st.subheader("Laporan Perbandingan Terakhir")
    st.json(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    st.title("🛰️ ForestWatch AI")
    st.caption(
        "Analisis Perbandingan Sentinel-1 vs Sentinel-2 untuk Perubahan Tutupan Hutan - Sesi 6"
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Perbandingan S1 vs S2", "Analisis Citra", "Perubahan Hutan", "Laporan"]
    )
    with tab1:
        render_demo()
    with tab2:
        render_upload()
    with tab3:
        render_perubahan()
    with tab4:
        render_laporan()


if __name__ == "__main__":
    main()
