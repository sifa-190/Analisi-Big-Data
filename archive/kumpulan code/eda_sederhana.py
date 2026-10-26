"""
EDA SEDERHANA - Taxi Kuning NYC (Jan 2019 - Jun 2020)

Setiap LANGKAH menjawab satu pertanyaan:
  1. Data ini isinya apa?
  2. Apakah datanya bersih? (lalu dibersihkan)
  3. Seperti apa angka-angka umumnya?
  4. Kapan paling ramai?
  5. Bagaimana tren dari bulan ke bulan?
  6. Di mana orderan paling banyak?
  7. Bagaimana soal uang dan cara bayar?
  8. Kesimpulan

Hasil: grafik PNG + ringkasan.txt di folder  hasil_eda_sederhana
"""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # grafik disimpan ke file
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# PENGATURAN (ubah di sini kalau lokasi file berbeda)
# ----------------------------------------------------------------------
FOLDER = Path(r"D:\Analisis Big Data\archive")
CSV_FILE = FOLDER / "yellow_tripdata_2019-2020_final.csv"
PARQUET = FOLDER / "trips_simple.parquet"   # versi bersih & cepat dibaca
OUT = FOLDER / "hasil_eda_sederhana"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
con = duckdb.connect()
temuan = []  # daftar kalimat kesimpulan


def langkah(nomor, judul):
    print("\n" + "=" * 65)
    print(f"LANGKAH {nomor}: {judul}")
    print("=" * 65)


def simpan_grafik(nama):
    plt.tight_layout()
    plt.savefig(OUT / nama, dpi=130)
    plt.close()
    print(f"  -> grafik disimpan: {nama}")


def catat(kalimat):
    print("  TEMUAN:", kalimat)
    temuan.append(kalimat)


def q(sql):
    return con.execute(sql).df()


# ======================================================================
# LANGKAH 1: DATA INI ISINYA APA?
# ======================================================================
langkah(1, "Mengenal data")

jumlah_mentah = q(f"SELECT COUNT(*) AS n FROM read_csv('{CSV_FILE.as_posix()}', "
                  f"header=true, sample_size=200000, ignore_errors=true)")["n"][0]
print(f"Jumlah baris data mentah : {jumlah_mentah:,} (1 baris = 1 perjalanan taksi)")
print("Kolom penting:")
print("  - tpep_pickup_datetime : waktu penumpang naik")
print("  - trip_distance        : jarak (mil)")
print("  - total_amount         : total bayar (dolar)")
print("  - PU_Zone / PU_Borough : zona & wilayah tempat naik")
print("  - payment_type         : cara bayar (1=kartu, 2=tunai)")

# ======================================================================
# LANGKAH 2: CEK KUALITAS DAN BERSIHKAN
# ======================================================================
langkah(2, "Cek kualitas data dan pembersihan")

# Pembersihan dilakukan SEKALI lalu disimpan sebagai Parquet agar cepat.
if not PARQUET.exists():
    print("Membersihkan data & menyimpan ke Parquet (sekali saja, bisa 10-30 menit)...")
    con.execute(f"""
        COPY (
            SELECT *,
                   date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 AS durasi_menit
            FROM read_csv('{CSV_FILE.as_posix()}', header=true,
                          sample_size=200000, ignore_errors=true)
            WHERE tpep_pickup_datetime >= '2019-01-01'      -- hanya periode yang dianalisis
              AND tpep_pickup_datetime <  '2020-07-01'
              AND trip_distance > 0                         -- jarak harus positif
              AND total_amount  > 0                         -- bayar harus positif
              AND date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime)
                  BETWEEN 60 AND 10800                      -- durasi 1 menit s/d 3 jam
        ) TO '{PARQUET.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

con.execute(f"""
    CREATE VIEW trips AS
    SELECT *, hour(tpep_pickup_datetime) AS jam,
           isodow(tpep_pickup_datetime) AS hari_ke,
           date_trunc('month', tpep_pickup_datetime) AS bulan
    FROM read_parquet('{PARQUET.as_posix()}')
""")

jumlah_bersih = q("SELECT COUNT(*) AS n FROM trips")["n"][0]
dibuang = jumlah_mentah - jumlah_bersih
print("Aturan pembersihan:")
print("  1) Tanggal di luar Jan 2019 - Jun 2020 dibuang (data salah catat)")
print("  2) Jarak <= 0 dibuang (perjalanan tidak valid)")
print("  3) Total bayar <= 0 dibuang (refund / error)")
print("  4) Durasi < 1 menit atau > 3 jam dibuang (tidak masuk akal)")
print(f"\nSebelum : {jumlah_mentah:,} baris")
print(f"Sesudah : {jumlah_bersih:,} baris")
print(f"Dibuang : {dibuang:,} baris ({dibuang / jumlah_mentah * 100:.2f}%)")

plt.figure(figsize=(6, 5))
plt.bar(["Data mentah", "Data bersih"], [jumlah_mentah, jumlah_bersih],
        color=["#c0c0c0", "#2e86ab"])
plt.title("Jumlah baris sebelum dan sesudah dibersihkan")
plt.ylabel("Jumlah perjalanan")
simpan_grafik("01_sebelum_sesudah_bersih.png")
catat(f"Dari {jumlah_mentah:,} perjalanan, {dibuang / jumlah_mentah * 100:.2f}% dibuang karena "
      f"tidak valid; {jumlah_bersih:,} perjalanan dipakai untuk analisis.")

# ======================================================================
# LANGKAH 3: ANGKA-ANGKA UMUM
# ======================================================================
langkah(3, "Statistik ringkas")

ringkas = q("""
    SELECT ROUND(AVG(trip_distance), 2)      AS rata2_jarak_mil,
           ROUND(median(trip_distance), 2)   AS median_jarak_mil,
           ROUND(AVG(durasi_menit), 1)       AS rata2_durasi_menit,
           ROUND(AVG(total_amount), 2)       AS rata2_bayar_dolar,
           ROUND(median(total_amount), 2)    AS median_bayar_dolar,
           ROUND(AVG(tip_amount), 2)         AS rata2_tip_dolar,
           ROUND(AVG(passenger_count), 2)    AS rata2_penumpang
    FROM trips
""").iloc[0]
print(ringkas.to_string())

# Histogram jarak & total bayar (dipotong persentil 99 agar mudah dibaca)
sampel = q("SELECT trip_distance, total_amount FROM trips USING SAMPLE reservoir(500000 ROWS) REPEATABLE (42)")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, kolom, judul_k, warna in [(ax[0], "trip_distance", "Jarak perjalanan (mil)", "#2e86ab"),
                                 (ax[1], "total_amount", "Total bayar (dolar)", "#f18f01")]:
    s = sampel[kolom]
    s = s[s <= s.quantile(0.99)]
    sns.histplot(s, bins=40, ax=a, color=warna)
    a.set_title(judul_k)
    a.set_xlabel("")
    a.set_ylabel("Jumlah perjalanan")
plt.suptitle("Sebagian besar perjalanan pendek dan murah", y=1.02)
simpan_grafik("02_distribusi_jarak_bayar.png")
catat(f"Perjalanan umumnya pendek: median {ringkas['median_jarak_mil']} mil, "
      f"rata-rata bayar ${ringkas['rata2_bayar_dolar']}, durasi sekitar "
      f"{ringkas['rata2_durasi_menit']} menit.")

# ======================================================================
# LANGKAH 4: KAPAN PALING RAMAI?
# ======================================================================
langkah(4, "Jam dan hari tersibuk")

per_jam = q("SELECT jam, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 1")
per_hari = q("SELECT hari_ke, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 1")
nama_hari = {1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 5: "Jumat", 6: "Sabtu", 7: "Minggu"}
per_hari["hari"] = per_hari["hari_ke"].map(nama_hari)

jam_ramai = per_jam.loc[per_jam["jml"].idxmax()]
jam_sepi = per_jam.loc[per_jam["jml"].idxmin()]
hari_ramai = per_hari.loc[per_hari["jml"].idxmax()]
hari_sepi = per_hari.loc[per_hari["jml"].idxmin()]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
warna_jam = ["#e4572e" if j == jam_ramai["jam"] else "#a8c5d6" for j in per_jam["jam"]]
ax[0].bar(per_jam["jam"], per_jam["jml"], color=warna_jam)
ax[0].set_title("Jumlah order per jam (merah = tersibuk)")
ax[0].set_xlabel("Jam")
ax[0].set_ylabel("Jumlah order")
warna_hari = ["#e4572e" if h == hari_ramai["hari"] else "#a8c5d6" for h in per_hari["hari"]]
ax[1].bar(per_hari["hari"], per_hari["jml"], color=warna_hari)
ax[1].set_title("Jumlah order per hari (merah = tersibuk)")
ax[1].set_ylabel("Jumlah order")
simpan_grafik("03_jam_dan_hari_tersibuk.png")

# Heatmap jam x hari
heat = q("SELECT hari_ke, jam, COUNT(*) AS jml FROM trips GROUP BY 1, 2")
pv = heat.pivot(index="hari_ke", columns="jam", values="jml").rename(index=nama_hari)
plt.figure(figsize=(13, 4.5))
sns.heatmap(pv, cmap="YlOrRd")
plt.title("Peta panas order: makin merah makin ramai")
plt.xlabel("Jam")
plt.ylabel("")
simpan_grafik("04_heatmap_jam_hari.png")

catat(f"Jam tersibuk pukul {int(jam_ramai['jam']):02d}:00 ({int(jam_ramai['jml']):,} order); "
      f"tersepi pukul {int(jam_sepi['jam']):02d}:00.")
catat(f"Hari tersibuk {hari_ramai['hari']}, tersepi {hari_sepi['hari']}.")

# ======================================================================
# LANGKAH 5: TREN BULANAN
# ======================================================================
langkah(5, "Tren order per bulan")

bulanan = q("SELECT bulan, COUNT(*) AS jml, SUM(total_amount) AS pendapatan FROM trips GROUP BY 1 ORDER BY 1")
bulanan["label"] = bulanan["bulan"].dt.strftime("%b %Y")
print(bulanan[["label", "jml", "pendapatan"]].to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(bulanan["label"], bulanan["jml"], marker="o", color="#2e86ab", linewidth=2)
plt.xticks(rotation=45)
plt.title("Jumlah order per bulan")
plt.ylabel("Jumlah order")
simpan_grafik("05_tren_bulanan.png")

# Bandingkan Feb 2020 (sebelum COVID) dengan bulan terendah
feb = bulanan[bulanan["bulan"] == "2020-02-01"]["jml"]
terendah = bulanan.loc[bulanan["jml"].idxmin()]
if len(feb):
    turun = (1 - terendah["jml"] / feb.iloc[0]) * 100
    catat(f"Order anjlok {turun:.0f}% dari Feb 2020 ke titik terendah ({terendah['label']}), "
          f"bertepatan dengan awal pandemi COVID-19.")

# ======================================================================
# LANGKAH 6: DI MANA PALING RAMAI?
# ======================================================================
langkah(6, "Zona dan wilayah terpopuler")

zona = q("""
    SELECT PU_Zone AS zona, COUNT(*) AS jml FROM trips
    WHERE PU_Zone IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10
""")
borough = q("""
    SELECT PU_Borough AS wilayah, COUNT(*) AS jml FROM trips
    WHERE PU_Borough IS NOT NULL GROUP BY 1 ORDER BY 2 DESC
""")
borough["persen"] = borough["jml"] / borough["jml"].sum() * 100
print(zona.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
sns.barplot(data=zona, x="jml", y="zona", color="#2e86ab", ax=ax[0])
ax[0].set_title("10 zona dengan penjemputan terbanyak")
ax[0].set_xlabel("Jumlah order")
ax[0].set_ylabel("")
ax[1].pie(borough["jml"], labels=borough["wilayah"], autopct="%1.1f%%", startangle=90)
ax[1].set_title("Pembagian order per wilayah")
simpan_grafik("06_zona_dan_wilayah.png")

catat(f"Zona terpopuler: {zona['zona'][0]}. Wilayah {borough['wilayah'][0]} menguasai "
      f"{borough['persen'][0]:.1f}% seluruh order.")

# ======================================================================
# LANGKAH 7: SOAL UANG DAN CARA BAYAR
# ======================================================================
langkah(7, "Pendapatan dan cara pembayaran")

uang_jam = q("SELECT jam, AVG(total_amount) AS rata2_bayar FROM trips GROUP BY 1 ORDER BY 1")
bayar = q("SELECT payment_type, COUNT(*) AS jml FROM trips GROUP BY 1 ORDER BY 2 DESC")
nama_bayar = {1: "Kartu kredit", 2: "Tunai", 3: "No charge", 4: "Dispute"}
bayar["metode"] = bayar["payment_type"].map(nama_bayar).fillna("Lainnya")
bayar["persen"] = bayar["jml"] / bayar["jml"].sum() * 100
tip = q("""
    SELECT ROUND(AVG(tip_amount / fare_amount) * 100, 1) AS tip_persen
    FROM trips WHERE payment_type = 1 AND fare_amount > 0
""")["tip_persen"][0]

jam_mahal = uang_jam.loc[uang_jam["rata2_bayar"].idxmax()]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(uang_jam["jam"], uang_jam["rata2_bayar"], marker="o", color="#e4572e", linewidth=2)
ax[0].set_title("Rata-rata bayar per perjalanan, per jam")
ax[0].set_xlabel("Jam")
ax[0].set_ylabel("Dolar")
ax[1].pie(bayar["jml"], labels=bayar["metode"], autopct="%1.1f%%", startangle=90)
ax[1].set_title("Cara pembayaran")
simpan_grafik("07_uang_dan_pembayaran.png")

catat(f"Rata-rata bayar tertinggi terjadi pukul {int(jam_mahal['jam']):02d}:00 "
      f"(${jam_mahal['rata2_bayar']:.2f}/perjalanan), belum tentu sama dengan jam tersibuk.")
catat(f"Cara bayar dominan: {bayar['metode'][0]} ({bayar['persen'][0]:.1f}%). "
      f"Rata-rata tip pembayar kartu {tip}% dari tarif.")

# ======================================================================
# LANGKAH 8: KESIMPULAN
# ======================================================================
langkah(8, "Kesimpulan")
for i, t in enumerate(temuan, 1):
    print(f"{i}. {t}")

with open(OUT / "ringkasan.txt", "w", encoding="utf-8") as f:
    f.write("RINGKASAN TEMUAN EDA - TAXI KUNING NYC\n\n")
    for i, t in enumerate(temuan, 1):
        f.write(f"{i}. {t}\n")

print(f"\nSelesai. Semua hasil ada di: {OUT}")

# ======================================================================
# CATATAN: KOLOM YANG DIPAKAI DI FILE INI
# ======================================================================
# Total 10 kolom asli + 4 kolom hasil hitung (dari 26 kolom di data gabungan).
#
# --- KOLOM ASLI (10) ---
# Kelompok KAPAN?
#   1. tpep_pickup_datetime  : waktu penumpang naik
#                              -> filter periode, diambil jam / hari / bulan
#   2. tpep_dropoff_datetime : waktu penumpang turun
#                              -> dikurangi waktu naik = durasi perjalanan
#
# Kelompok DI MANA?
#   3. PU_Zone               : nama zona tempat penumpang naik  -> top 10 zona
#   4. PU_Borough            : nama wilayah tempat naik         -> pembagian per wilayah
#
# Kelompok BERAPA UANGNYA?
#   5. total_amount          : total bayar -> filter (> 0), rata-rata bayar,
#                              pendapatan per bulan, bayar per jam
#   6. fare_amount           : tarif dasar -> pembagi untuk persentase tip
#   7. tip_amount            : tip         -> rata-rata tip
#   8. payment_type          : cara bayar  -> kartu vs tunai, filter tip (kartu saja)
#
# Kelompok SEPERTI APA PERJALANANNYA?
#   9. trip_distance         : jarak (mil) -> filter (> 0), rata-rata & median jarak
#  10. passenger_count       : jumlah penumpang -> rata-rata penumpang
#
# --- KOLOM HASIL HITUNG (4) ---
#   - durasi_menit : (tpep_dropoff_datetime - tpep_pickup_datetime) dalam menit
#   - jam          : jam dari tpep_pickup_datetime (0-23)
#   - hari_ke      : hari dari tpep_pickup_datetime (1=Senin ... 7=Minggu)
#   - bulan        : bulan dari tpep_pickup_datetime
#
# --- KOLOM YANG TIDAK DIPAKAI DI FILE INI ---
#   VendorID, RatecodeID, store_and_fwd_flag, PULocationID, DOLocationID,
#   extra, mta_tax, tolls_amount, improvement_surcharge, congestion_surcharge,
#   PU_LocationID, PU_service_zone, DO_LocationID, DO_Borough, DO_Zone,
#   DO_service_zone
#   (DO_Zone, PULocationID, DOLocationID baru dipakai di dashboard.py)