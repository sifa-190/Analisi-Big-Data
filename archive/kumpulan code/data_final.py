from pathlib import Path
import pandas as pd

folder = Path(r"D:\Analisis Big Data\archive")

# Pola ini hanya cocok dengan yellow_tripdata_YYYY-MM.csv,
# jadi file hasil gabungan (..._final.csv) tidak ikut terbaca
files = sorted(folder.glob("yellow_tripdata_20[0-9][0-9]-[0-9][0-9].csv"))
output = folder / "yellow_tripdata_2019-2020_final.csv"

# Tabel lookup zona (dipakai untuk pickup & dropoff)
zone = pd.read_csv(folder / "taxi+_zone_lookup.csv")
pu = zone.add_prefix("PU_")
do = zone.add_prefix("DO_")

print(f"Ditemukan {len(files)} file trip")

first = True
total_rows = 0
for f in files:
    print("Memproses:", f.name)
    for chunk in pd.read_csv(f, chunksize=500_000, low_memory=False):
        chunk = (chunk
                 .merge(pu, left_on="PULocationID", right_on="PU_LocationID", how="left")
                 .merge(do, left_on="DOLocationID", right_on="DO_LocationID", how="left"))
        chunk.to_csv(output, mode="w" if first else "a",
                     header=first, index=False)
        first = False
        total_rows += len(chunk)

print(f"Selesai. Total {total_rows:,} baris -> {output}")