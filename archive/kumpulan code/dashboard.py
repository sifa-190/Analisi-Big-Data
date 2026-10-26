from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ======================================================================
# KONFIGURASI
# ======================================================================
FOLDER = Path(r"D:\Analisis Big Data\archive")
CSV_FILE = FOLDER / "yellow_tripdata_2019-2020_final.csv"
PARQUET_FILE = FOLDER / "trips_clean.parquet"

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

tab_waktu, tab_lokasi, tab_uang, tab_ops, tab_tren = st.tabs(
    ["⏰ Waktu", "📍 Lokasi", "💰 Pendapatan & Tip", "🚦 Efisiensi", "📈 Tren Bulanan"]
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