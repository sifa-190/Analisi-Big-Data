import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

# Library machine learning. Kalau belum terpasang, dashboard tetap jalan
# dan tab terkait menampilkan petunjuk instalasi.
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# Library peta (membaca shapefile batas zona)
try:
    import geopandas as gpd
    GEOPANDAS_OK = True
except ImportError:
    GEOPANDAS_OK = False

# ======================================================================
# KONFIGURASI
# ======================================================================
FOLDER = Path(r"D:\Analisis Big Data\archive")
CSV_FILE = FOLDER / "yellow_tripdata_2019-2020_final.csv"
PARQUET_FILE = FOLDER / "trips_clean.parquet"
TAXI_ZONES_DIR = FOLDER / "taxi_zones"      # folder shapefile batas zona

HARI = {1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 5: "Jumat", 6: "Sabtu", 7: "Minggu"}
PAYMENT = {1: "Kartu kredit", 2: "Tunai", 3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Void"}

st.set_page_config(page_title="Dashboard Taxi NYC", page_icon="🚕", layout="wide")


# ======================================================================
# DATA: CSV (11 GB) dikonversi SEKALI ke Parquet yang sudah dibersihkan
# ======================================================================
@st.cache_resource(show_spinner=False)
def get_connection():
    con = duckdb.connect()
    if not PARQUET_FILE.exists():
        with st.spinner("Pertama kali: mengubah CSV ke Parquet (bisa 10-30 menit, hanya sekali)..."):
            con.execute(f"""
                COPY (
                    SELECT *,
                           date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 AS durasi_menit
                    FROM read_csv('{CSV_FILE.as_posix()}', header=true,
                                  sample_size=200000, ignore_errors=true)
                    WHERE tpep_pickup_datetime >= '2019-01-01'
                      AND tpep_pickup_datetime <  '2020-07-01'
                      AND trip_distance > 0
                      AND total_amount > 0
                      AND date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime)
                          BETWEEN 60 AND 10800
                ) TO '{PARQUET_FILE.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{PARQUET_FILE.as_posix()}')")
    return con


con = get_connection()


@st.cache_data(show_spinner=False)
def run(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


@st.cache_data(show_spinner=False)
def get_options():
    borough = run("SELECT DISTINCT PU_Borough FROM trips WHERE PU_Borough IS NOT NULL ORDER BY 1")
    return borough["PU_Borough"].tolist()


# ======================================================================
# PETA: baca shapefile zona -> GeoJSON, lalu gambar peta berwarna (choropleth)
# ======================================================================
@st.cache_data(show_spinner=False)
def muat_geojson():
    """Membaca batas zona taksi dari shapefile dan mengubahnya ke GeoJSON (lat/lon).
    Mengembalikan (geojson, pesan_error). Salah satunya selalu None."""
    if not GEOPANDAS_OK:
        return None, "Library geopandas belum terpasang. Jalankan: python -m pip install geopandas"
    daftar_shp = sorted(TAXI_ZONES_DIR.rglob("*.shp"))
    if not daftar_shp:
        return None, f"File .shp tidak ditemukan di folder: {TAXI_ZONES_DIR}"

    gdf = gpd.read_file(daftar_shp[0])
    if gdf.crs is None:                      # shapefile TLC memakai koordinat NY State Plane (EPSG:2263)
        gdf = gdf.set_crs(epsg=2263)
    gdf = gdf.to_crs(epsg=4326)              # ubah ke lat/lon supaya bisa digambar di peta web

    kolom_id = next((c for c in gdf.columns if c.lower() == "locationid"), None)
    if kolom_id is None:
        return None, "Kolom LocationID tidak ditemukan di shapefile."
    gdf["LocationID"] = gdf[kolom_id].astype(int)

    # Beberapa zona terdiri dari beberapa poligon -> gabungkan per LocationID
    gdf = gdf[["LocationID", "geometry"]].dissolve(by="LocationID").reset_index()
    gdf["geometry"] = gdf.geometry.simplify(0.0003)   # sederhanakan bentuk agar ringan
    return json.loads(gdf.to_json()), None


def gambar_peta(df, kolom_warna, judul, diskrit=False, label_warna=None):
    """Menggambar peta zona. df wajib punya kolom: location_id, zona, serta kolom_warna."""
    geojson, galat = muat_geojson()
    if galat:
        st.warning(f"Peta belum bisa ditampilkan. {galat}")
        return
    opsi = dict(
        geojson=geojson,
        locations="location_id",
        featureidkey="properties.LocationID",
        color=kolom_warna,
        hover_name="zona",
        opacity=0.75,
        zoom=9.6,
        center={"lat": 40.73, "lon": -73.95},
        title=judul,
        labels={kolom_warna: label_warna or kolom_warna},
    )
    if not diskrit:
        opsi["color_continuous_scale"] = "YlOrRd"
    # plotly baru memakai choropleth_map, versi lama choropleth_mapbox
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(df, map_style="carto-positron", **opsi)
    else:
        fig = px.choropleth_mapbox(df, mapbox_style="carto-positron", **opsi)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=620)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# SIDEBAR: FILTER
# ======================================================================
st.sidebar.title("🚕 Filter")
rentang = st.sidebar.date_input(
    "Rentang tanggal",
    value=(date(2019, 1, 1), date(2020, 6, 30)),
    min_value=date(2019, 1, 1),
    max_value=date(2020, 6, 30),
)
if not isinstance(rentang, (tuple, list)) or len(rentang) != 2:
    st.info("Pilih tanggal awal dan akhir di sidebar.")
    st.stop()

semua_borough = get_options()
borough_pilih = st.sidebar.multiselect("Borough pickup", semua_borough, default=semua_borough)
top_n = st.sidebar.slider("Jumlah Top-N (zona/rute)", 5, 25, 10)

if not borough_pilih:
    st.warning("Pilih minimal satu borough.")
    st.stop()

mulai, akhir = rentang
akhir_eks = akhir + timedelta(days=1)
borough_sql = ", ".join("'" + b.replace("'", "''") + "'" for b in borough_pilih)
WHERE = (
    f"tpep_pickup_datetime >= '{mulai}' AND tpep_pickup_datetime < '{akhir_eks}' "
    f"AND PU_Borough IN ({borough_sql})"
)

# ======================================================================
# HEADER + KPI
# ======================================================================
st.title("Dashboard Analisis Taxi Kuning NYC")
st.caption(f"Periode {mulai} s/d {akhir} • Borough: {', '.join(borough_pilih)}")

kpi = run(f"""
    SELECT COUNT(*) AS orders,
           SUM(total_amount) AS revenue,
           AVG(total_amount) AS avg_fare,
           AVG(trip_distance) AS avg_dist,
           SUM(total_amount) / SUM(durasi_menit) AS rev_per_min
    FROM trips WHERE {WHERE}
""").iloc[0]

if kpi["orders"] == 0 or pd.isna(kpi["orders"]):
    st.warning("Tidak ada data untuk filter ini.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total order", f"{int(kpi['orders']):,}")
c2.metric("Total pendapatan", f"${kpi['revenue']:,.0f}")
c3.metric("Rata-rata per trip", f"${kpi['avg_fare']:.2f}")
c4.metric("Rata-rata jarak", f"{kpi['avg_dist']:.2f} mil")
c5.metric("Pendapatan / menit", f"${kpi['rev_per_min']:.2f}")

(tab_waktu, tab_lokasi, tab_harian, tab_peta, tab_cluster,
 tab_uang, tab_ops, tab_tren) = st.tabs(
    ["⏰ Waktu", "📍 Lokasi", "💵 Penghasilan Harian", "🗺️ Peta",
     "🤖 Clustering Zona", "💰 Pendapatan & Tip", "🚦 Efisiensi", "📈 Tren Bulanan"]
)

# ======================================================================
# TAB WAKTU
# ======================================================================
with tab_waktu:
    heat = run(f"""
        SELECT isodow(tpep_pickup_datetime) AS d,
               hour(tpep_pickup_datetime)   AS jam,
               COUNT(*)                     AS jml_order,
               AVG(total_amount)            AS avg_pendapatan
        FROM trips WHERE {WHERE} GROUP BY 1, 2
    """)
    heat["hari"] = heat["d"].map(HARI)
    urutan = list(HARI.values())

    kiri, kanan = st.columns(2)
    with kiri:
        pv = heat.pivot(index="hari", columns="jam", values="jml_order").reindex(urutan)
        fig = px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(x="Jam", y="Hari", color="Order"),
                        title="Jumlah order: jam × hari")
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        pv2 = heat.pivot(index="hari", columns="jam", values="avg_pendapatan").reindex(urutan)
        fig = px.imshow(pv2, aspect="auto", color_continuous_scale="Viridis",
                        labels=dict(x="Jam", y="Hari", color="$ / trip"),
                        title="Rata-rata pendapatan per trip: jam × hari")
        st.plotly_chart(fig, use_container_width=True)

    per_jam = run(f"""
        SELECT hour(tpep_pickup_datetime) AS jam,
               COUNT(*) AS jml_order,
               AVG(total_amount) AS avg_pendapatan,
               SUM(total_amount) / SUM(durasi_menit) AS rev_per_min
        FROM trips WHERE {WHERE} GROUP BY 1 ORDER BY 1
    """)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=per_jam["jam"], y=per_jam["jml_order"], name="Jumlah order", opacity=0.6)
    fig.add_scatter(x=per_jam["jam"], y=per_jam["avg_pendapatan"], name="Rata-rata $/trip",
                    mode="lines+markers", secondary_y=True)
    fig.update_layout(title="Order vs rata-rata pendapatan per jam", xaxis_title="Jam")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Jam paling ramai belum tentu jam paling menguntungkan. "
               "Bandingkan batang (volume) dengan garis (nilai per trip).")

# ======================================================================
# TAB LOKASI
# ======================================================================
with tab_lokasi:
    kiri, kanan = st.columns(2)
    zona = run(f"""
        SELECT PU_Zone AS zona, PU_Borough AS borough,
               COUNT(*) AS jml_order, ROUND(SUM(total_amount)) AS total_pendapatan
        FROM trips WHERE {WHERE} AND PU_Zone IS NOT NULL
        GROUP BY 1, 2 ORDER BY total_pendapatan DESC LIMIT {top_n}
    """)
    with kiri:
        fig = px.bar(zona.sort_values("total_pendapatan"), x="total_pendapatan", y="zona",
                     color="borough", orientation="h",
                     title=f"Top {top_n} zona pickup berdasarkan pendapatan")
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        zona_o = run(f"""
            SELECT PU_Zone AS zona, PU_Borough AS borough, COUNT(*) AS jml_order
            FROM trips WHERE {WHERE} AND PU_Zone IS NOT NULL
            GROUP BY 1, 2 ORDER BY jml_order DESC LIMIT {top_n}
        """)
        fig = px.bar(zona_o.sort_values("jml_order"), x="jml_order", y="zona",
                     color="borough", orientation="h",
                     title=f"Top {top_n} zona pickup berdasarkan jumlah order")
        st.plotly_chart(fig, use_container_width=True)

    kiri, kanan = st.columns(2)
    with kiri:
        bor = run(f"""
            SELECT PU_Borough AS borough, COUNT(*) AS jml_order,
                   SUM(total_amount) AS pendapatan, AVG(total_amount) AS avg_fare
            FROM trips WHERE {WHERE} GROUP BY 1 ORDER BY 2 DESC
        """)
        fig = px.pie(bor, names="borough", values="jml_order", hole=0.45,
                     title="Proporsi order per borough")
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        rute = run(f"""
            SELECT PU_Zone || ' → ' || DO_Zone AS rute,
                   COUNT(*) AS jml_order, ROUND(AVG(total_amount), 2) AS avg_fare
            FROM trips WHERE {WHERE} AND PU_Zone IS NOT NULL AND DO_Zone IS NOT NULL
            GROUP BY 1 ORDER BY jml_order DESC LIMIT {top_n}
        """)
        fig = px.bar(rute.sort_values("jml_order"), x="jml_order", y="rute",
                     orientation="h", hover_data=["avg_fare"],
                     title=f"Top {top_n} rute terpopuler")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat tabel detail zona & borough"):
        st.dataframe(zona, use_container_width=True)
        st.dataframe(bor, use_container_width=True)

# ======================================================================
# TAB PENGHASILAN HARIAN  (BARU)
# ======================================================================
with tab_harian:
    # Satu baris per tanggal: jumlah order dan total uang yang dibayar penumpang
    harian = run(f"""
        SELECT CAST(tpep_pickup_datetime AS DATE) AS tanggal,
               COUNT(*)          AS jml_order,
               SUM(total_amount) AS penghasilan,
               AVG(total_amount) AS avg_bayar
        FROM trips WHERE {WHERE}
        GROUP BY 1 ORDER BY 1
    """)
    harian["tanggal"] = pd.to_datetime(harian["tanggal"])
    # Rata-rata bergerak 7 hari: meredam naik-turun harian supaya tren terlihat
    harian["rata2_7_hari"] = harian["penghasilan"].rolling(7, min_periods=1).mean()

    hari_max = harian.loc[harian["penghasilan"].idxmax()]
    hari_min = harian.loc[harian["penghasilan"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rata-rata penghasilan / hari", f"${harian['penghasilan'].mean():,.0f}")
    k2.metric("Hari tertinggi", f"${hari_max['penghasilan']:,.0f}",
              hari_max["tanggal"].strftime("%d %b %Y"), delta_color="off")
    k3.metric("Hari terendah", f"${hari_min['penghasilan']:,.0f}",
              hari_min["tanggal"].strftime("%d %b %Y"), delta_color="off")
    k4.metric("Jumlah hari", f"{len(harian):,}")

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_scatter(x=harian["tanggal"], y=harian["penghasilan"], name="Penghasilan harian",
                    mode="lines", line=dict(width=1), opacity=0.5)
    fig.add_scatter(x=harian["tanggal"], y=harian["rata2_7_hari"], name="Rata-rata 7 hari",
                    mode="lines", line=dict(width=3))
    fig.update_layout(title="Penghasilan per hari (dolar)", yaxis_title="Dolar",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    kiri, kanan = st.columns(2)
    with kiri:
        # Rata-rata penghasilan menurut nama hari (Senin ... Minggu)
        harian["hari"] = (harian["tanggal"].dt.dayofweek + 1).map(HARI)
        per_nama_hari = (harian.groupby("hari")["penghasilan"].mean()
                         .reindex(list(HARI.values())).reset_index())
        fig = px.bar(per_nama_hari, x="hari", y="penghasilan",
                     title="Rata-rata penghasilan per hari dalam seminggu",
                     labels={"penghasilan": "Dolar", "hari": ""})
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        fig = px.line(harian, x="tanggal", y="jml_order",
                      title="Jumlah order per hari", labels={"jml_order": "Order", "tanggal": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Penghasilan di sini adalah total uang yang dibayar penumpang (kolom total_amount, "
               "termasuk pajak dan tip kartu), bukan pendapatan bersih sopir. "
               "Penurunan tajam sejak Maret 2020 adalah dampak COVID-19.")
    with st.expander("Lihat tabel penghasilan per hari"):
        st.dataframe(harian[["tanggal", "jml_order", "penghasilan", "avg_bayar"]]
                     .sort_values("tanggal", ascending=False), use_container_width=True)

# ======================================================================
# TAB PETA  (BARU)
# ======================================================================
with tab_peta:
    st.markdown("Peta zona taksi New York. **Makin gelap warnanya, makin tinggi nilainya.** "
                "Arahkan kursor ke zona untuk melihat detail.")

    data_peta = run(f"""
        SELECT PULocationID AS location_id,
               PU_Zone AS zona, PU_Borough AS borough,
               COUNT(*)          AS jml_order,
               SUM(total_amount) AS total_pendapatan,
               AVG(total_amount) AS avg_bayar
        FROM trips WHERE {WHERE} AND PU_Zone IS NOT NULL
        GROUP BY 1, 2, 3
    """)
    data_peta["location_id"] = data_peta["location_id"].astype(int)

    pilihan = st.radio(
        "Warnai peta berdasarkan",
        ["Jumlah order", "Total pendapatan", "Rata-rata bayar per trip"],
        horizontal=True,
    )
    kolom_peta = {"Jumlah order": "jml_order",
                  "Total pendapatan": "total_pendapatan",
                  "Rata-rata bayar per trip": "avg_bayar"}[pilihan]

    # Beberapa zona (Midtown, bandara) nilainya jauh lebih besar dari zona lain,
    # sehingga zona lain tampak pucat semua. Skala log membantu membedakannya.
    pakai_log = st.checkbox("Pakai skala logaritma (supaya zona yang sepi tetap terlihat)")
    if pakai_log:
        data_peta["nilai_log"] = np.log10(data_peta[kolom_peta].clip(lower=1))
        gambar_peta(data_peta, "nilai_log", f"{pilihan} per zona jemput (skala log10)",
                    label_warna="log10")
    else:
        gambar_peta(data_peta, kolom_peta, f"{pilihan} per zona jemput")

# ======================================================================
# TAB CLUSTERING ZONA  (BARU)
# ======================================================================
with tab_cluster:
    st.markdown("""
**Tujuan:** mengelompokkan zona taksi yang *karakternya mirip* secara otomatis, tanpa label dari kita
(*unsupervised learning*). Hasilnya membantu perusahaan menyusun strategi per kelompok zona, bukan per zona satu-satu.

**Algoritma:** K-Means. Setiap zona dijadikan satu titik berdasarkan 6 ciri (jumlah order, rata-rata bayar,
jarak, durasi, porsi order malam, porsi order akhir pekan). K-Means lalu mengumpulkan titik-titik yang berdekatan
ke dalam *k* kelompok.
""")

    if not SKLEARN_OK:
        st.error("Library scikit-learn belum terpasang. Jalankan: "
                 "python -m pip install scikit-learn")
    else:
        min_order = st.slider("Minimal jumlah order per zona (zona yang terlalu sepi diabaikan)",
                              100, 5000, 1000, step=100)

        # ------------------------------------------------------------
        # LANGKAH 1: Bangun tabel fitur. SATU BARIS = SATU ZONA JEMPUT.
        # ------------------------------------------------------------
        fitur_df = run(f"""
            SELECT PULocationID AS location_id,
                   PU_Zone      AS zona,
                   PU_Borough   AS borough,
                   COUNT(*)            AS jml_order,
                   AVG(total_amount)   AS avg_bayar,
                   AVG(trip_distance)  AS avg_jarak,
                   AVG(durasi_menit)   AS avg_durasi,
                   -- persentase order yang terjadi malam hari (20.00 - 05.59)
                   AVG(CASE WHEN hour(tpep_pickup_datetime) >= 20
                             OR hour(tpep_pickup_datetime) < 6 THEN 1.0 ELSE 0.0 END) * 100 AS persen_malam,
                   -- persentase order yang terjadi Sabtu/Minggu
                   AVG(CASE WHEN isodow(tpep_pickup_datetime) >= 6 THEN 1.0 ELSE 0.0 END) * 100 AS persen_weekend
            FROM trips
            WHERE {WHERE} AND PU_Zone IS NOT NULL
            GROUP BY 1, 2, 3
            HAVING COUNT(*) >= {min_order}
        """)
        fitur_df["location_id"] = fitur_df["location_id"].astype(int)

        if len(fitur_df) < 8:
            st.warning("Zona yang memenuhi syarat terlalu sedikit untuk di-cluster. "
                       "Turunkan 'minimal jumlah order' atau pilih lebih banyak borough di sidebar.")
        else:
            # Nama kolom -> nama yang mudah dibaca di grafik
            FITUR = {
                "jml_order": "Jumlah order",
                "avg_bayar": "Rata-rata bayar ($)",
                "avg_jarak": "Rata-rata jarak (mil)",
                "avg_durasi": "Rata-rata durasi (menit)",
                "persen_malam": "% order malam",
                "persen_weekend": "% order akhir pekan",
            }
            kolom_fitur = list(FITUR.keys())

            # ------------------------------------------------------------
            # LANGKAH 2: Siapkan data untuk K-Means
            #  a) log1p pada jumlah order: beberapa zona (Midtown) jauh lebih ramai dari yang lain.
            #     Tanpa log, hasil clustering hanya ditentukan oleh jumlah order.
            #  b) StandardScaler: menyamakan skala semua fitur (rata-rata 0, simpangan baku 1).
            #     K-Means memakai jarak, jadi fitur berskala besar (order: ribuan) akan
            #     mengalahkan fitur berskala kecil (jarak: beberapa mil) kalau tidak disamakan.
            # ------------------------------------------------------------
            X = fitur_df[kolom_fitur].copy()
            X["jml_order"] = np.log1p(X["jml_order"])
            X_skala = StandardScaler().fit_transform(X)

            # ------------------------------------------------------------
            # LANGKAH 3: Pilih jumlah cluster (k).
            #  Silhouette score (-1 s/d 1): makin tinggi = zona makin cocok dengan kelompoknya
            #  dan makin berbeda dari kelompok lain.
            #  Inertia (elbow): total jarak titik ke pusat kelompoknya; makin kecil makin rapat.
            #  Dicoba k = 2 sampai 8, lalu k dengan silhouette tertinggi dijadikan saran.
            # ------------------------------------------------------------
            hasil_k = []
            for k_uji in range(2, 9):
                km = KMeans(n_clusters=k_uji, n_init=10, random_state=42).fit(X_skala)
                hasil_k.append({"k": k_uji,
                                "silhouette": silhouette_score(X_skala, km.labels_),
                                "inertia": km.inertia_})
            hasil_k = pd.DataFrame(hasil_k)
            k_saran = int(hasil_k.loc[hasil_k["silhouette"].idxmax(), "k"])

            kiri, kanan = st.columns(2)
            with kiri:
                fig = px.line(hasil_k, x="k", y="silhouette", markers=True,
                              title="Silhouette score (makin tinggi makin baik)")
                st.plotly_chart(fig, use_container_width=True)
            with kanan:
                fig = px.line(hasil_k, x="k", y="inertia", markers=True,
                              title="Elbow / inertia (cari titik tekukan)")
                st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Saran jumlah cluster berdasarkan silhouette tertinggi: k = {k_saran}. "
                       "Kamu bebas mengubahnya di bawah, pilih yang paling mudah ditafsirkan.")

            k = st.slider("Jumlah cluster (k)", 2, 8, k_saran)

            # ------------------------------------------------------------
            # LANGKAH 4: Jalankan K-Means dengan k pilihan.
            #  random_state tetap supaya hasilnya sama setiap kali dijalankan.
            # ------------------------------------------------------------
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            label = model.fit_predict(X_skala)
            fitur_df["cluster"] = ["Cluster " + str(i + 1) for i in label]

            # ------------------------------------------------------------
            # LANGKAH 5: Tafsirkan tiap cluster.
            #  Pusat cluster dalam satuan z-score: nilai positif = di atas rata-rata semua zona,
            #  negatif = di bawah. Dua fitur dengan nilai paling ekstrem dijadikan "ciri utama".
            # ------------------------------------------------------------
            z_profil = pd.DataFrame(X_skala, columns=kolom_fitur).groupby(label).mean()

            def ciri_utama(idx):
                z = z_profil.loc[idx].sort_values(key=np.abs, ascending=False).head(2)
                return ", ".join(f"{FITUR[n]} {'tinggi' if v > 0 else 'rendah'}" for n, v in z.items())

            profil = fitur_df.groupby("cluster").agg(
                jumlah_zona=("zona", "count"),
                rata2_order=("jml_order", "mean"),
                rata2_bayar=("avg_bayar", "mean"),
                rata2_jarak=("avg_jarak", "mean"),
                rata2_durasi=("avg_durasi", "mean"),
                persen_malam=("persen_malam", "mean"),
                persen_weekend=("persen_weekend", "mean"),
            ).round(2)
            profil["ciri_utama"] = [ciri_utama(int(c.split()[1]) - 1) for c in profil.index]
            # Tiga zona paling ramai per cluster sebagai contoh yang mudah dikenali
            contoh = (fitur_df.sort_values("jml_order", ascending=False)
                      .groupby("cluster")["zona"].apply(lambda s: ", ".join(s.head(3))))
            profil["contoh_zona"] = contoh

            st.subheader("Profil tiap cluster")
            st.dataframe(profil, use_container_width=True)

            # Heatmap z-score: merah = di atas rata-rata, biru = di bawah rata-rata
            z_tampil = z_profil.copy()
            z_tampil.index = ["Cluster " + str(i + 1) for i in z_tampil.index]
            z_tampil.columns = [FITUR[c] for c in z_tampil.columns]
            fig = px.imshow(z_tampil, color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                            aspect="auto", text_auto=".1f",
                            title="Karakter tiap cluster (merah = di atas rata-rata, biru = di bawah)")
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------
            # LANGKAH 6: Visualisasi.
            #  PCA meringkas 6 fitur menjadi 2 sumbu agar bisa digambar di bidang datar.
            #  Zona yang berdekatan di grafik = karakternya mirip.
            # ------------------------------------------------------------
            pca = PCA(n_components=2, random_state=42)
            koordinat = pca.fit_transform(X_skala)
            fitur_df["pc1"], fitur_df["pc2"] = koordinat[:, 0], koordinat[:, 1]
            urutan_cluster = sorted(fitur_df["cluster"].unique())

            kiri, kanan = st.columns(2)
            with kiri:
                fig = px.scatter(fitur_df, x="pc1", y="pc2", color="cluster",
                                 hover_name="zona",
                                 hover_data={"borough": True, "jml_order": ":,", "pc1": False, "pc2": False},
                                 category_orders={"cluster": urutan_cluster},
                                 title="Sebaran zona (hasil PCA 2 dimensi)",
                                 labels={"pc1": "Sumbu 1", "pc2": "Sumbu 2"})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Dua sumbu ini menjelaskan {pca.explained_variance_ratio_.sum() * 100:.0f}% "
                           "keragaman data.")
            with kanan:
                jml_per_cluster = fitur_df["cluster"].value_counts().reindex(urutan_cluster).reset_index()
                jml_per_cluster.columns = ["cluster", "jumlah_zona"]
                fig = px.bar(jml_per_cluster, x="cluster", y="jumlah_zona", color="cluster",
                             category_orders={"cluster": urutan_cluster},
                             title="Jumlah zona per cluster")
                st.plotly_chart(fig, use_container_width=True)

            # Peta hasil clustering: warna = kelompok
            st.subheader("Peta hasil clustering")
            gambar_peta(fitur_df, "cluster", "Zona jemput berdasarkan cluster",
                        diskrit=True, label_warna="Cluster")

            with st.expander("Lihat daftar zona per cluster"):
                st.dataframe(
                    fitur_df[["cluster", "zona", "borough"] + kolom_fitur]
                    .sort_values(["cluster", "jml_order"], ascending=[True, False])
                    .round(2), use_container_width=True)

            st.info("**Cara menjelaskan:** K-Means tidak diberi tahu mana zona bisnis atau permukiman. "
                    "Algoritma hanya melihat kemiripan angka. Nama dan makna tiap cluster kita tafsirkan sendiri "
                    "dari profilnya, lalu dipakai untuk strategi, misalnya cluster dengan order tinggi dan "
                    "porsi malam tinggi perlu armada lebih banyak pada malam hari.")

# ======================================================================
# TAB PENDAPATAN & TIP
# ======================================================================
with tab_uang:
    kiri, kanan = st.columns(2)
    with kiri:
        fig = px.line(per_jam, x="jam", y="rev_per_min", markers=True,
                      title="Pendapatan per menit perjalanan, per jam",
                      labels={"rev_per_min": "$ / menit", "jam": "Jam"})
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        tip = run(f"""
            SELECT hour(tpep_pickup_datetime) AS jam,
                   AVG(tip_amount / NULLIF(fare_amount, 0)) * 100 AS tip_persen
            FROM trips WHERE {WHERE} AND payment_type = 1 AND fare_amount > 0
            GROUP BY 1 ORDER BY 1
        """)
        fig = px.line(tip, x="jam", y="tip_persen", markers=True,
                      title="Rata-rata tip (%) per jam, kartu kredit saja",
                      labels={"tip_persen": "Tip (%)", "jam": "Jam"})
        st.plotly_chart(fig, use_container_width=True)

    bayar = run(f"""
        SELECT date_trunc('month', tpep_pickup_datetime) AS bulan,
               payment_type, COUNT(*) AS jml
        FROM trips WHERE {WHERE} GROUP BY 1, 2 ORDER BY 1
    """)
    bayar["metode"] = bayar["payment_type"].map(PAYMENT).fillna("Lainnya")
    fig = px.bar(bayar, x="bulan", y="jml", color="metode", barmode="relative",
                 title="Komposisi metode pembayaran per bulan")
    fig.update_layout(barnorm="percent", yaxis_title="Persen")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip tunai tidak tercatat di data, jadi analisis tip hanya memakai pembayaran kartu.")

# ======================================================================
# TAB EFISIENSI
# ======================================================================
with tab_ops:
    speed = run(f"""
        SELECT hour(tpep_pickup_datetime) AS jam,
               AVG(trip_distance / (durasi_menit / 60.0)) AS mph
        FROM trips
        WHERE {WHERE} AND trip_distance / (durasi_menit / 60.0) < 80
        GROUP BY 1 ORDER BY 1
    """)
    kiri, kanan = st.columns(2)
    with kiri:
        fig = px.line(speed, x="jam", y="mph", markers=True,
                      title="Kecepatan rata-rata per jam (indikator kemacetan)",
                      labels={"mph": "Mil / jam", "jam": "Jam"})
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        intra = run(f"""
            SELECT CASE WHEN PULocationID = DOLocationID THEN 'Dalam zona yang sama'
                        ELSE 'Antar zona' END AS jenis,
                   COUNT(*) AS jml
            FROM trips WHERE {WHERE} GROUP BY 1
        """)
        fig = px.pie(intra, names="jenis", values="jml", hole=0.45,
                     title="Perjalanan dalam zona vs antar zona")
        st.plotly_chart(fig, use_container_width=True)

    pax = run(f"""
        SELECT passenger_count AS penumpang, COUNT(*) AS jml
        FROM trips WHERE {WHERE} AND passenger_count BETWEEN 1 AND 6
        GROUP BY 1 ORDER BY 1
    """)
    fig = px.bar(pax, x="penumpang", y="jml", title="Distribusi jumlah penumpang per trip")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB TREN BULANAN
# ======================================================================
with tab_tren:
    bulanan = run(f"""
        SELECT date_trunc('month', tpep_pickup_datetime) AS bulan,
               COUNT(*) AS jml_order, SUM(total_amount) AS pendapatan
        FROM trips WHERE {WHERE} GROUP BY 1 ORDER BY 1
    """)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=bulanan["bulan"], y=bulanan["jml_order"], name="Jumlah order", opacity=0.6)
    fig.add_scatter(x=bulanan["bulan"], y=bulanan["pendapatan"], name="Pendapatan ($)",
                    mode="lines+markers", secondary_y=True)
    fig.update_layout(title="Tren order dan pendapatan per bulan")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Perhatikan penurunan tajam sejak Maret 2020 (dampak COVID-19).")
    st.dataframe(bulanan, use_container_width=True)