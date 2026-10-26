"""
EDA Taxi Kuning NYC 2019-2020 (data besar, diproses dengan DuckDB)

Alur:
  1. Ubah CSV mentah -> Parquet mentah (sekali saja, TANPA filter apa pun)
  2. Gambaran umum, missing value, kualitas data
  3. Statistik deskriptif & outlier
  4. Distribusi (sampel 1 juta baris), kategorikal, waktu, korelasi
Semua grafik & tabel disimpan di folder eda_output/
"""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # simpan gambar ke file, tidak membuka jendela
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ======================================================================
# KONFIGURASI
# ======================================================================
FOLDER = Path(r"D:\Analisis Big Data\archive")
CSV_FILE = FOLDER / "yellow_tripdata_2019-2020_final.csv"
RAW_PARQUET = FOLDER / "trips_raw.parquet"
OUT = FOLDER / "eda_output"
OUT.mkdir(exist_ok=True)

UKURAN_SAMPEL = 1_000_000
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


def simpan(nama):
    plt.tight_layout()
    plt.savefig(OUT / nama, dpi=130)
    plt.close()
    print("  grafik disimpan:", nama)


# ======================================================================
# 1. SIAPKAN DATA
# ======================================================================
con = duckdb.connect()

if not RAW_PARQUET.exists():
    print("Mengubah CSV ke Parquet mentah (sekali saja, bisa 10-30 menit)...")
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv('{CSV_FILE.as_posix()}', header=true,
                                   sample_size=200000, ignore_errors=true)
        ) TO '{RAW_PARQUET.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

con.execute(f"""
    CREATE VIEW trips AS
    SELECT *,
           date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 AS durasi_menit,
           hour(tpep_pickup_datetime)      AS jam,
           isodow(tpep_pickup_datetime)    AS hari_ke,
           date_trunc('month', tpep_pickup_datetime) AS bulan
    FROM read_parquet('{RAW_PARQUET.as_posix()}')
""")


def q(sql):
    return con.execute(sql).df()


# ======================================================================
# 2. GAMBARAN UMUM
# ======================================================================
judul("2A. GAMBARAN UMUM")
n_baris = q("SELECT COUNT(*) AS n FROM trips")["n"][0]
print(f"Jumlah baris : {n_baris:,}")

skema = q("DESCRIBE SELECT * FROM read_parquet('%s')" % RAW_PARQUET.as_posix())
skema = skema[["column_name", "column_type"]]
print(f"Jumlah kolom : {len(skema)}\n")
print(skema.to_string(index=False))

rentang = q("SELECT MIN(tpep_pickup_datetime) AS awal, MAX(tpep_pickup_datetime) AS akhir FROM trips")
print("\nRentang pickup di data:", rentang["awal"][0], "->", rentang["akhir"][0])

cols = skema["column_name"].tolist()
num_cols = [c for c, t in zip(skema["column_name"], skema["column_type"])
            if t in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT", "TINYINT", "HUGEINT")
            or t.startswith("DECIMAL")]

# ======================================================================
# 2B. MISSING VALUE
# ======================================================================
judul("2B. MISSING VALUE")
sql_null = ", ".join(f"SUM(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END) AS \"{c}\"" for c in cols)
miss = q(f"SELECT {sql_null} FROM trips").T.reset_index()
miss.columns = ["kolom", "jumlah_null"]
miss["persen"] = (miss["jumlah_null"] / n_baris * 100).round(2)
miss = miss.sort_values("persen", ascending=False)
print(miss.to_string(index=False))
miss.to_csv(OUT / "missing_value.csv", index=False)

mm = miss[miss["jumlah_null"] > 0]
if len(mm):
    plt.figure(figsize=(9, max(3, 0.35 * len(mm))))
    sns.barplot(data=mm, x="persen", y="kolom", color="#e4572e")
    plt.title("Persentase missing value per kolom")
    plt.xlabel("% null")
    simpan("01_missing_value.png")

# ======================================================================
# 2C. KUALITAS DATA (aturan bisnis)
# ======================================================================
judul("2C. KUALITAS DATA")
kualitas = q("""
    SELECT
      SUM(CASE WHEN tpep_pickup_datetime < '2019-01-01' OR tpep_pickup_datetime >= '2020-07-01' THEN 1 ELSE 0 END) AS tanggal_di_luar_periode,
      SUM(CASE WHEN durasi_menit <= 0 THEN 1 ELSE 0 END)               AS durasi_nol_atau_negatif,
      SUM(CASE WHEN durasi_menit > 180 THEN 1 ELSE 0 END)              AS durasi_lebih_3_jam,
      SUM(CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END)              AS jarak_nol_atau_negatif,
      SUM(CASE WHEN trip_distance > 100 THEN 1 ELSE 0 END)             AS jarak_lebih_100_mil,
      SUM(CASE WHEN passenger_count = 0 THEN 1 ELSE 0 END)             AS penumpang_nol,
      SUM(CASE WHEN passenger_count > 6 THEN 1 ELSE 0 END)             AS penumpang_lebih_6,
      SUM(CASE WHEN fare_amount <= 0 THEN 1 ELSE 0 END)                AS fare_nol_atau_negatif,
      SUM(CASE WHEN total_amount <= 0 THEN 1 ELSE 0 END)               AS total_nol_atau_negatif,
      SUM(CASE WHEN tip_amount < 0 THEN 1 ELSE 0 END)                  AS tip_negatif,
      SUM(CASE WHEN total_amount > 500 THEN 1 ELSE 0 END)              AS total_lebih_500_dolar,
      SUM(CASE WHEN durasi_menit > 0
                AND trip_distance / (durasi_menit / 60.0) > 80 THEN 1 ELSE 0 END) AS kecepatan_lebih_80mph,
      SUM(CASE WHEN tpep_dropoff_datetime < tpep_pickup_datetime THEN 1 ELSE 0 END) AS dropoff_sebelum_pickup
    FROM trips
""").T.reset_index()
kualitas.columns = ["masalah", "jumlah"]
kualitas["persen"] = (kualitas["jumlah"] / n_baris * 100).round(3)
print(kualitas.to_string(index=False))
kualitas.to_csv(OUT / "kualitas_data.csv", index=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=kualitas.sort_values("persen"), x="persen", y="masalah", color="#f18f01")
plt.title("Persentase baris bermasalah per aturan")
plt.xlabel("% dari seluruh baris")
simpan("02_kualitas_data.png")

# ======================================================================
# 3. STATISTIK DESKRIPTIF & OUTLIER
# ======================================================================
judul("3. STATISTIK DESKRIPTIF (data penuh)")
stat_cols = [c for c in num_cols if c not in ("PU_LocationID", "DO_LocationID")] + ["durasi_menit"]
baris_stat = []
for c in stat_cols:
    r = q(f"""
        SELECT '{c}' AS kolom, COUNT("{c}") AS n,
               ROUND(AVG("{c}"), 3) AS rata2, ROUND(STDDEV("{c}"), 3) AS std,
               MIN("{c}") AS min,
               ROUND(approx_quantile("{c}", 0.25), 3) AS q25,
               ROUND(approx_quantile("{c}", 0.50), 3) AS median,
               ROUND(approx_quantile("{c}", 0.75), 3) AS q75,
               ROUND(approx_quantile("{c}", 0.99), 3) AS p99,
               MAX("{c}") AS max
        FROM trips
    """)
    baris_stat.append(r)
stat = pd.concat(baris_stat, ignore_index=True)
print(stat.to_string(index=False))
stat.to_csv(OUT / "statistik_deskriptif.csv", index=False)

judul("3B. OUTLIER (metode IQR) pada kolom utama")
outlier_rows = []
for c in ["trip_distance", "fare_amount", "tip_amount", "total_amount", "durasi_menit"]:
    r = q(f"""
        WITH b AS (
            SELECT approx_quantile("{c}", 0.25) AS q1, approx_quantile("{c}", 0.75) AS q3 FROM trips
        )
        SELECT '{c}' AS kolom,
               ROUND(q1 - 1.5 * (q3 - q1), 2) AS batas_bawah,
               ROUND(q3 + 1.5 * (q3 - q1), 2) AS batas_atas,
               SUM(CASE WHEN "{c}" < q1 - 1.5 * (q3 - q1) OR "{c}" > q3 + 1.5 * (q3 - q1)
                        THEN 1 ELSE 0 END) AS jumlah_outlier
        FROM trips, b GROUP BY q1, q3
    """)
    outlier_rows.append(r)
outlier = pd.concat(outlier_rows, ignore_index=True)
outlier["persen"] = (outlier["jumlah_outlier"] / n_baris * 100).round(2)
print(outlier.to_string(index=False))
outlier.to_csv(OUT / "outlier_iqr.csv", index=False)

# ======================================================================
# 4. VISUALISASI (sampel acak)
# ======================================================================
judul(f"4. VISUALISASI (sampel {UKURAN_SAMPEL:,} baris, seed tetap)")
df = q(f"SELECT * FROM trips USING SAMPLE reservoir({UKURAN_SAMPEL} ROWS) REPEATABLE (42)")
print("Sampel dimuat:", df.shape)

# Pembersihan ringan KHUSUS untuk grafik (statistik di atas tetap memakai data mentah)
bersih = df[(df["trip_distance"] > 0) & (df["total_amount"] > 0)
            & (df["durasi_menit"].between(1, 180))
            & (df["tpep_pickup_datetime"] >= "2019-01-01")
            & (df["tpep_pickup_datetime"] < "2020-07-01")].copy()
print(f"Baris sampel setelah pembersihan ringan: {len(bersih):,} dari {len(df):,}")

# 4A. Histogram (dipotong di persentil 99 supaya bentuk distribusi terlihat)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a, (c, judul_c) in zip(ax.ravel(), [
    ("trip_distance", "Jarak (mil)"), ("durasi_menit", "Durasi (menit)"),
    ("fare_amount", "Tarif dasar ($)"), ("tip_amount", "Tip ($)"),
    ("total_amount", "Total bayar ($)"), ("passenger_count", "Jumlah penumpang"),
]):
    s = bersih[c].dropna()
    s = s[s <= s.quantile(0.99)]
    sns.histplot(s, bins=50, ax=a, color="#2e86ab")
    a.set_title(judul_c)
    a.set_xlabel("")
plt.suptitle("Distribusi variabel numerik (dipotong di persentil 99)", y=1.02)
simpan("03_histogram.png")

# 4B. Boxplot
fig, ax = plt.subplots(1, 4, figsize=(16, 5))
for a, c in zip(ax, ["trip_distance", "fare_amount", "tip_amount", "durasi_menit"]):
    s = bersih[c]
    sns.boxplot(y=s[s <= s.quantile(0.99)], ax=a, color="#a3c4bc")
    a.set_title(c)
    a.set_ylabel("")
plt.suptitle("Boxplot (dipotong di persentil 99)", y=1.02)
simpan("04_boxplot.png")

# 4C. Kategorikal
judul("4C. KATEGORIKAL (data penuh)")
kategori = {
    "payment_type": {1: "Kartu", 2: "Tunai", 3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Void"},
    "RatecodeID": {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau/Westchester", 5: "Negotiated", 6: "Group"},
    "VendorID": {1: "Creative Mobile", 2: "VeriFone"},
    "store_and_fwd_flag": None,
    "PU_Borough": None,
}
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax = ax.ravel()
for i, (c, peta) in enumerate(kategori.items()):
    t = q(f'SELECT "{c}" AS nilai, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 2 DESC')
    t["persen"] = (t["jml"] / n_baris * 100).round(2)
    if peta:
        t["nilai"] = t["nilai"].map(lambda x: peta.get(x, f"Kode {x}"))
    print(f"\n{c}:")
    print(t.head(10).to_string(index=False))
    sns.barplot(data=t.head(10), x="jml", y=t.head(10)["nilai"].astype(str), ax=ax[i], color="#6a4c93")
    ax[i].set_title(c)
    ax[i].set_ylabel("")
    ax[i].set_xlabel("jumlah")

top_zona = q("""
    SELECT PU_Zone AS zona, COUNT(*) AS jml FROM trips
    WHERE PU_Zone IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10
""")
sns.barplot(data=top_zona, x="jml", y="zona", ax=ax[5], color="#1982c4")
ax[5].set_title("Top 10 zona pickup")
ax[5].set_ylabel("")
plt.suptitle("Variabel kategorikal", y=1.01)
simpan("05_kategorikal.png")

# 4D. Waktu
judul("4D. POLA WAKTU (data penuh)")
bulanan = q("SELECT bulan, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 1")
print("Bulan yang muncul di data (cek apakah ada bulan di luar periode):")
print(bulanan.to_string(index=False))
bulanan.to_csv(OUT / "order_per_bulan_mentah.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
dalam = bulanan[(bulanan["bulan"] >= "2019-01-01") & (bulanan["bulan"] < "2020-07-01")]
sns.lineplot(data=dalam, x="bulan", y="jml", marker="o", ax=ax[0])
ax[0].set_title("Order per bulan (periode 2019-01 s/d 2020-06)")
ax[0].tick_params(axis="x", rotation=45)

per_jam = q("SELECT jam, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 1")
sns.barplot(data=per_jam, x="jam", y="jml", ax=ax[1], color="#2e86ab")
ax[1].set_title("Order per jam")

nama_hari = {1: "Sen", 2: "Sel", 3: "Rab", 4: "Kam", 5: "Jum", 6: "Sab", 7: "Min"}
per_hari = q("SELECT hari_ke, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 1")
per_hari["hari"] = per_hari["hari_ke"].map(nama_hari)
sns.barplot(data=per_hari, x="hari", y="jml", ax=ax[2], color="#f18f01")
ax[2].set_title("Order per hari")
simpan("06_pola_waktu.png")

heat = q("""
    SELECT hari_ke, jam, COUNT(*) AS jml FROM trips
    WHERE tpep_pickup_datetime >= '2019-01-01' AND tpep_pickup_datetime < '2020-07-01'
    GROUP BY 1, 2
""")
pv = heat.pivot(index="hari_ke", columns="jam", values="jml").rename(index=nama_hari)
plt.figure(figsize=(13, 4.5))
sns.heatmap(pv, cmap="YlOrRd")
plt.title("Heatmap order: hari × jam")
plt.ylabel("")
simpan("07_heatmap_hari_jam.png")

# 4E. Korelasi
judul("4E. KORELASI (sampel bersih)")
kolom_korelasi = ["passenger_count", "trip_distance", "durasi_menit", "fare_amount",
                  "extra", "mta_tax", "tip_amount", "tolls_amount", "total_amount"]
kolom_korelasi = [c for c in kolom_korelasi if c in bersih.columns]
korelasi = bersih[kolom_korelasi].corr().round(2)
print(korelasi.to_string())
korelasi.to_csv(OUT / "korelasi.csv")
plt.figure(figsize=(9, 7))
sns.heatmap(korelasi, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Korelasi antar variabel numerik")
simpan("08_korelasi.png")

# 4F. Hubungan jarak - tarif
plt.figure(figsize=(8, 6))
s = bersih.sample(min(50_000, len(bersih)), random_state=1)
s = s[(s["trip_distance"] < 30) & (s["fare_amount"] < 120)]
sns.scatterplot(data=s, x="trip_distance", y="fare_amount", alpha=0.15, s=8)
plt.title("Jarak vs tarif dasar (sampel 50 ribu)")
simpan("09_jarak_vs_tarif.png")

# ======================================================================
# RINGKASAN
# ======================================================================
judul("SELESAI")
print(f"Semua output tersimpan di: {OUT}")
for f in sorted(OUT.iterdir()):
    print(" -", f.name)