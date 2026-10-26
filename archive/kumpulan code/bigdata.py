import duckdb

f = r"D:/Analisis Big Data/archive/yellow_tripdata_2019-2020_final.csv"

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW trips AS
    SELECT *,
           date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 AS durasi_menit
    FROM read_csv_auto('{f}')
    WHERE tpep_pickup_datetime >= '2019-01-01'
      AND tpep_pickup_datetime <  '2020-07-01'
      AND trip_distance > 0
      AND total_amount > 0
      AND date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) BETWEEN 60 AND 10800
""")

# 1. Heatmap jam x hari: jumlah order dan rata-rata pendapatan
heat = con.execute("""
    SELECT dayname(tpep_pickup_datetime) AS hari,
           hour(tpep_pickup_datetime)    AS jam,
           COUNT(*)                      AS jml_order,
           ROUND(AVG(total_amount), 2)   AS avg_pendapatan
    FROM trips GROUP BY 1, 2 ORDER BY 1, 2
""").df()

# 2. Top 10 zona pickup berdasarkan total pendapatan
zona = con.execute("""
    SELECT PU_Zone, PU_Borough,
           COUNT(*) AS jml_order,
           ROUND(SUM(total_amount)) AS total_pendapatan
    FROM trips GROUP BY 1, 2 ORDER BY total_pendapatan DESC LIMIT 10
""").df()

# 3. Rata-rata persentase tip per jam (kartu kredit saja)
tip = con.execute("""
    SELECT hour(tpep_pickup_datetime) AS jam,
           ROUND(AVG(tip_amount / NULLIF(fare_amount, 0)) * 100, 1) AS tip_persen
    FROM trips WHERE payment_type = 1 GROUP BY 1 ORDER BY 1
""").df()

# 4. Kecepatan rata-rata per jam (indikator kemacetan)
speed = con.execute("""
    SELECT hour(tpep_pickup_datetime) AS jam,
           ROUND(AVG(trip_distance / (durasi_menit / 60.0)), 1) AS mph
    FROM trips GROUP BY 1 ORDER BY 1
""").df()

# 5. Tren bulanan
bulanan = con.execute("""
    SELECT date_trunc('month', tpep_pickup_datetime) AS bulan,
           COUNT(*) AS jml_order,
           ROUND(SUM(total_amount)) AS pendapatan
    FROM trips GROUP BY 1 ORDER BY 1
""").df()

print(heat.head(), zona, tip, speed, bulanan, sep="\n\n")