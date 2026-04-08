import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'data', 'olist.db')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'customer_features.csv')


def build_features():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)

    # ── Step 1: RFM + core features ─────────────────────
    print("Step 1/4 — Building RFM and core features...")
    rfm_query = """
    SELECT
        c.customer_unique_id,

        CAST(
            JULIANDAY('2018-09-01') -
            JULIANDAY(MAX(o.order_purchase_timestamp))
        AS INTEGER)                             AS recency_days,

        COUNT(DISTINCT o.order_id)             AS frequency,
        ROUND(SUM(oi.price), 2)                AS monetary,
        ROUND(AVG(oi.freight_value), 2)        AS avg_freight_paid,
        ROUND(AVG(oi_count.items_count), 2)    AS avg_items_per_order,

        ROUND(AVG(r.review_score), 2)          AS avg_review_score,
        MIN(r.review_score)                    AS min_review_score,

        ROUND(AVG(
            CASE WHEN o.order_delivered_customer_date
                      > o.order_estimated_delivery_date
                 THEN 1.0 ELSE 0.0 END
        ) * 100, 2)                            AS late_delivery_pct,

        SUM(
            CASE WHEN o.order_delivered_customer_date
                      > o.order_estimated_delivery_date
                 THEN 1 ELSE 0 END
        )                                      AS late_delivery_count,

        ROUND(AVG(p.payment_installments), 2) AS avg_installments,

        CASE
            WHEN JULIANDAY('2018-09-01') -
                 JULIANDAY(MAX(o.order_purchase_timestamp)) > 90
            THEN 1 ELSE 0
        END                                    AS is_churned

    FROM customers c
    JOIN orders o
      ON c.customer_id = o.customer_id
      AND o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    JOIN order_items oi
      ON o.order_id = oi.order_id
    LEFT JOIN reviews r
      ON o.order_id = r.order_id
    LEFT JOIN payments p
      ON o.order_id = p.order_id
    LEFT JOIN (
        SELECT order_id, COUNT(*) AS items_count
        FROM order_items
        GROUP BY order_id
    ) oi_count ON o.order_id = oi_count.order_id
    GROUP BY c.customer_unique_id
    """

    df = pd.read_sql(rfm_query, conn)
    print(f"   → {len(df):,} customers loaded")

    # ── Step 2: Category diversity ───────────────────────
    print("Step 2/4 — Calculating category diversity...")
    cat_diversity_query = """
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT ct.product_category_name_english) AS category_diversity
    FROM customers c
    JOIN orders o
      ON c.customer_id = o.customer_id
      AND o.order_status = 'delivered'
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p     ON oi.product_id = p.product_id
    JOIN category_translation ct
      ON p.product_category_name = ct.product_category_name
    GROUP BY c.customer_unique_id
    """
    df_diversity = pd.read_sql(cat_diversity_query, conn)
    df = df.merge(df_diversity, on='customer_unique_id', how='left')
    print(f"   → Done")

    # ── Step 3: Top category (done in pandas, not SQL) ───
    print("Step 3/4 — Finding top category per customer...")
    cat_query = """
    SELECT
        c.customer_unique_id,
        ct.product_category_name_english AS category
    FROM customers c
    JOIN orders o
      ON c.customer_id = o.customer_id
      AND o.order_status = 'delivered'
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p     ON oi.product_id = p.product_id
    JOIN category_translation ct
      ON p.product_category_name = ct.product_category_name
    """
    df_cats = pd.read_sql(cat_query, conn)

    top_cat = (
        df_cats
        .groupby(['customer_unique_id', 'category'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .drop_duplicates(subset='customer_unique_id')
        [['customer_unique_id', 'category']]
        .rename(columns={'category': 'top_category'})
    )
    df = df.merge(top_cat, on='customer_unique_id', how='left')
    print(f"   → Done")

    conn.close()

    # ── Step 4: Clean and save ───────────────────────────
    print("Step 4/4 — Cleaning and saving...")

    df = df.assign(
        avg_review_score=df['avg_review_score'].fillna(df['avg_review_score'].median()),
        min_review_score=df['min_review_score'].fillna(df['min_review_score'].median()),
        late_delivery_pct=df['late_delivery_pct'].fillna(0),
        late_delivery_count=df['late_delivery_count'].fillna(0),
        category_diversity=df['category_diversity'].fillna(1),
        avg_installments=df['avg_installments'].fillna(1),
        top_category=df['top_category'].fillna('unknown'),
    )

    df.dropna(subset=['monetary', 'recency_days'], inplace=True)

    df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Feature table saved → data/customer_features.csv")
    print(f"   Total customers  : {len(df):,}")
    print(f"   Churned (1)      : {df['is_churned'].sum():,}  ({df['is_churned'].mean()*100:.1f}%)")
    print(f"   Active  (0)      : {(df['is_churned']==0).sum():,}  ({(df['is_churned']==0).mean()*100:.1f}%)")
    print(f"   Total features   : {df.shape[1]}")
    print(f"\n── Sample output ──")
    print(df[['customer_unique_id', 'recency_days', 'frequency',
              'monetary', 'avg_review_score', 'late_delivery_pct',
              'category_diversity', 'top_category', 'is_churned'
              ]].head(5).to_string(index=False))

    return df


if __name__ == '__main__':
    build_features()