"""Two small audits for the final report. Reads data, writes two small csv files in reports/.

1. Share of each rating band inside each split (train / validation / test).
2. Which products are in the raw file but not in the clean file, and why.

Run from the project root:   python src/audit_extras.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "digikala-comments.csv"
CLEAN = ROOT / "data" / "interim" / "comments_clean.parquet"
PRODUCT_SPLIT = ROOT / "data" / "interim" / "product_split.parquet"
REPORTS = ROOT / "reports"

con = duckdb.connect()

# ---------------------------------------------------------------- audit 1
cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{CLEAN}')").fetchall()]
print("columns of comments_clean:", cols)

if "rate_band" in cols:
    # the interim file may use low / mid / high; the project text says low / medium / high
    band = "CASE WHEN c.rate_band = 'mid' THEN 'medium' ELSE c.rate_band END"
else:
    r = "TRY_CAST(c.rate AS DOUBLE)"
    band = (f"CASE WHEN {r} >= 1 AND {r} < 2.5 THEN 'low' WHEN {r} >= 2.5 AND {r} < 4.0 THEN 'medium' "
            f"WHEN {r} >= 4.0 AND {r} <= 5.0 THEN 'high' ELSE NULL END")

shares = con.sql(f"""
    SELECT s.split,
           COALESCE({band}, '(no band)') AS rating_band,
           COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY s.split), 2) AS pct_of_split
    FROM read_parquet('{CLEAN}') c
    JOIN read_parquet('{PRODUCT_SPLIT}') s USING (product_id)
    GROUP BY s.split, rating_band
    ORDER BY CASE s.split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END, rating_band
""").df()
print("\nRating-band share per split (rows of the clean file; '(no band)' = rate 0 or outside 1 to 5):")
print(shares.to_string(index=False))
print("\nrows per split:")
print(shares.groupby("split")["n"].sum().to_string())
shares.to_csv(REPORTS / "rating_band_share_per_split.csv", index=False)

# ---------------------------------------------------------------- audit 2
raw_products = f"SELECT DISTINCT TRY_CAST(product_id AS BIGINT) AS product_id FROM read_csv('{RAW}', all_varchar=true) WHERE product_id IS NOT NULL"
clean_products = f"SELECT DISTINCT CAST(product_id AS BIGINT) AS product_id FROM read_parquet('{CLEAN}')"
n_raw = con.sql(f"SELECT COUNT(*) FROM ({raw_products})").fetchall()[0][0]
n_clean = con.sql(f"SELECT COUNT(*) FROM ({clean_products})").fetchall()[0][0]
print(f"\nproducts in raw file: {n_raw:,} | in clean file: {n_clean:,} | difference: {n_raw - n_clean:,}")

missing = con.sql(f"""
    WITH miss AS (SELECT product_id FROM ({raw_products}) EXCEPT SELECT product_id FROM ({clean_products})),
         raw AS (SELECT *, TRY_CAST(product_id AS BIGINT) AS pid FROM read_csv('{RAW}', all_varchar=true)),
         id_stats AS (SELECT id, COUNT(*) AS id_rows_in_raw, COUNT(DISTINCT body) AS distinct_bodies_for_id
                      FROM raw GROUP BY id)
    SELECT r.pid AS product_id, r.id, i.id_rows_in_raw, i.distinct_bodies_for_id,
           r.rate, r.recommendation_status, left(r.body, 60) AS body_start
    FROM raw r JOIN miss m ON r.pid = m.product_id JOIN id_stats i USING (id)
    ORDER BY r.pid, r.id
    LIMIT 200
""").df()
print("\nraw rows of the products that are missing from the clean file (at most 200 shown):")
print(missing.to_string(index=False))
print("\nreading help: id_rows_in_raw > 1 means the id appears more than once in the raw file;")
print("distinct_bodies_for_id > 1 means those copies have different review texts (a conflicting id).")
missing.to_csv(REPORTS / "missing_products_check.csv", index=False)
