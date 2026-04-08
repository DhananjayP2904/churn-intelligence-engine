import pandas as pd
import sqlite3
import os
import shutil

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DB_PATH   = os.path.join(BASE_DIR, 'data', 'olist.db')
SCORES    = os.path.join(BASE_DIR, 'data', 'customer_scores.csv')
BRIEFS    = os.path.join(BASE_DIR, 'data', 'ai_churn_briefs.csv')

# ── Change this to your actual OneDrive folder path ───
ONEDRIVE  = r"C:/Users/Dhananjay Pandagre/OneDrive"


def get_conn():
    return sqlite3.connect(DB_PATH)


# ════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ════════════════════════════════════════════════════════
def build_executive_overview():
    conn = get_conn()

    df = pd.read_sql("""
        SELECT
            strftime('%Y', o.order_purchase_timestamp)       AS year,
            strftime('%m', o.order_purchase_timestamp)       AS month,
            strftime('%Y-%m', o.order_purchase_timestamp)    AS year_month,

            COUNT(DISTINCT o.order_id)                       AS total_orders,
            COUNT(DISTINCT c.customer_unique_id)             AS unique_customers,
            ROUND(SUM(oi.price), 2)                          AS total_revenue,
            ROUND(AVG(oi.price), 2)                          AS avg_order_value,
            ROUND(SUM(oi.freight_value), 2)                  AS total_freight,

            -- Late delivery rate
            ROUND(AVG(
                CASE WHEN o.order_delivered_customer_date
                          > o.order_estimated_delivery_date
                     THEN 1.0 ELSE 0.0 END
            ) * 100, 2)                                      AS late_delivery_pct,

            -- Avg review score
            ROUND(AVG(r.review_score), 2)                    AS avg_review_score,

            -- Cancelled orders
            SUM(CASE WHEN o.order_status = 'canceled'
                     THEN 1 ELSE 0 END)                      AS cancelled_orders

        FROM orders o
        JOIN customers c   ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status IN ('delivered', 'canceled')
        GROUP BY year_month
        ORDER BY year_month
    """, conn)

    conn.close()
    return df


# ════════════════════════════════════════════════════════
# PAGE 2 — Sales & Product Insights
# ════════════════════════════════════════════════════════
def build_sales_product():
    conn = get_conn()

    df = pd.read_sql("""
        SELECT
            ct.product_category_name_english                 AS category,
            strftime('%Y-%m', o.order_purchase_timestamp)    AS year_month,
            o.order_status,

            COUNT(DISTINCT o.order_id)                       AS total_orders,
            COUNT(DISTINCT c.customer_unique_id)             AS unique_customers,
            ROUND(SUM(oi.price), 2)                          AS revenue,
            ROUND(AVG(oi.price), 2)                          AS avg_price,
            ROUND(SUM(oi.freight_value), 2)                  AS total_freight,
            ROUND(AVG(oi.freight_value), 2)                  AS avg_freight,
            COUNT(oi.order_item_id)                          AS units_sold,
            ROUND(AVG(r.review_score), 2)                    AS avg_review,

            -- Late delivery per category
            ROUND(AVG(
                CASE WHEN o.order_delivered_customer_date
                          > o.order_estimated_delivery_date
                     THEN 1.0 ELSE 0.0 END
            ) * 100, 2)                                      AS late_delivery_pct,

            -- Payment type (most common per group)
            pay.payment_type,
            ROUND(AVG(pay.payment_installments), 1)          AS avg_installments

        FROM orders o
        JOIN customers c    ON o.customer_id = c.customer_id
        JOIN order_items oi  ON o.order_id = oi.order_id
        JOIN products p      ON oi.product_id = p.product_id
        JOIN category_translation ct
                             ON p.product_category_name = ct.product_category_name
        LEFT JOIN reviews r  ON o.order_id = r.order_id
        LEFT JOIN payments pay ON o.order_id = pay.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY category, year_month, pay.payment_type
        ORDER BY revenue DESC
    """, conn)

    conn.close()
    return df


# ════════════════════════════════════════════════════════
# PAGE 3 — Customer Behavior & Segmentation
# ════════════════════════════════════════════════════════
def build_customer_behavior():
    conn = get_conn()

    # Base customer data
    df = pd.read_sql("""
        SELECT
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,

            COUNT(DISTINCT o.order_id)                       AS frequency,
            ROUND(SUM(oi.price + oi.freight_value), 2)       AS lifetime_value,
            ROUND(AVG(oi.price), 2)                          AS avg_order_value,
            MIN(o.order_purchase_timestamp)                  AS first_order_date,
            MAX(o.order_purchase_timestamp)                  AS last_order_date,

            CAST(JULIANDAY('2018-09-01') -
                 JULIANDAY(MAX(o.order_purchase_timestamp))
            AS INTEGER)                                      AS recency_days,

            ROUND(AVG(r.review_score), 2)                    AS avg_review_score,

            ROUND(AVG(
                CASE WHEN o.order_delivered_customer_date
                          > o.order_estimated_delivery_date
                     THEN 1.0 ELSE 0.0 END
            ) * 100, 2)                                      AS late_delivery_pct,

            pay.payment_type                                 AS preferred_payment,
            ROUND(AVG(pay.payment_installments), 1)          AS avg_installments,
            COUNT(DISTINCT ct.product_category_name_english) AS category_diversity

        FROM customers c
        JOIN orders o     ON c.customer_id = o.customer_id
                         AND o.order_status = 'delivered'
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p     ON oi.product_id = p.product_id
        JOIN category_translation ct
                            ON p.product_category_name = ct.product_category_name
        LEFT JOIN reviews r  ON o.order_id = r.order_id
        LEFT JOIN payments pay ON o.order_id = pay.order_id
        GROUP BY c.customer_unique_id, c.customer_city,
                 c.customer_state, pay.payment_type
    """, conn)

    conn.close()

    # RFM Segments — add in pandas
    df['rfm_segment'] = 'One-Time Buyer'
    df.loc[
        (df['frequency'] >= 2) & (df['recency_days'] <= 180),
        'rfm_segment'
    ] = 'Loyal Customer'
    df.loc[
        (df['frequency'] >= 2) & (df['recency_days'] > 180),
        'rfm_segment'
    ] = 'At-Risk Loyal'
    df.loc[
        (df['lifetime_value'] >= df['lifetime_value'].quantile(0.9)),
        'rfm_segment'
    ] = 'High Value'

    # Spending tier
    df['spend_tier'] = pd.cut(
        df['lifetime_value'],
        bins   = [0, 50, 150, 500, 99999],
        labels = ['Low (<$50)', 'Mid ($50-150)',
                  'High ($150-500)', 'Premium (>$500)']
    )

    return df


# ════════════════════════════════════════════════════════
# PAGE 4 — Churn Intelligence
# ════════════════════════════════════════════════════════
def build_churn_intelligence():
    # Load model scores
    scores = pd.read_csv(SCORES)

    # Load AI briefs
    briefs = pd.read_csv(BRIEFS)

    # Merge
    df = scores.merge(
        briefs[['customer_unique_id', 'ai_brief']],
        on  = 'customer_unique_id',
        how = 'left'
    )

    df['ai_brief'].fillna('Low risk — no brief generated', inplace=True)

    # Revenue at risk per segment
    df['revenue_at_risk'] = df.apply(
        lambda r: r['monetary'] if r['churn_risk_score'] >= 65 else 0,
        axis=1
    )

    # Keep only columns Power BI needs
    keep = [
        'customer_unique_id', 'frequency', 'monetary',
        'avg_review_score', 'min_review_score',
        'late_delivery_pct', 'late_delivery_count',
        'category_diversity', 'avg_installments',
        'avg_items_per_order', 'avg_freight_paid',
        'top_category', 'churn_risk_score',
        'risk_segment', 'is_churned',
        'revenue_at_risk', 'ai_brief'
    ]

    return df[keep]


# ════════════════════════════════════════════════════════
# UPLOAD TO ONEDRIVE
# ════════════════════════════════════════════════════════
def upload_to_onedrive(files_dict):
    os.makedirs(ONEDRIVE, exist_ok=True)

    for filename, df in files_dict.items():
        dest = os.path.join(ONEDRIVE, filename)
        df.to_csv(dest, index=False)
        print(f"   ✅ {filename} → {dest}  ({len(df):,} rows)")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def export_all():
    print("Building all 4 dashboard datasets...\n")

    datasets = {
        "page1_executive_overview.csv"    : build_executive_overview(),
        "page2_sales_product.csv"         : build_sales_product(),
        "page3_customer_behavior.csv"     : build_customer_behavior(),
        "page4_churn_intelligence.csv"    : build_churn_intelligence(),
    }

    for name, df in datasets.items():
        print(f"   {name}: {len(df):,} rows, {df.shape[1]} columns")

    print("\nUploading to OneDrive...")
    upload_to_onedrive(datasets)

    print("\n✅ All 4 files in OneDrive — refresh Power BI to update dashboard")


if __name__ == '__main__':
    export_all()