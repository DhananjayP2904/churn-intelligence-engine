import pandas as pd
import sqlite3
import os

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
DB_PATH     = os.path.join(BASE_DIR, 'data', 'olist.db')

CSV_FILES = {
    'olist_customers_dataset.csv'           : 'customers',
    'olist_orders_dataset.csv'              : 'orders',
    'olist_order_items_dataset.csv'         : 'order_items',
    'olist_order_payments_dataset.csv'      : 'payments',
    'olist_order_reviews_dataset.csv'       : 'reviews',
    'olist_products_dataset.csv'            : 'products',
    'olist_sellers_dataset.csv'             : 'sellers',
    'olist_geolocation_dataset.csv'         : 'geolocation',
    'product_category_name_translation.csv' : 'category_translation',
}

# Columns that contain dates — we will standardise all of them
DATE_COLUMNS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'shipping_limit_date',
    'review_creation_date',
    'review_answer_timestamp',
]

def parse_dates(df):
    """Convert any date column found in df to YYYY-MM-DD HH:MM:SS format."""
    for col in DATE_COLUMNS:
        if col in df.columns:
            # Try both common formats — dayfirst handles DD-MM-YYYY
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def load_csvs_to_sqlite():
    conn = sqlite3.connect(DB_PATH)

    for filename, table_name in CSV_FILES.items():
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f'Loading {filename} ...')
        df = pd.read_csv(filepath)
        df = parse_dates(df)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f'  → {len(df):,} rows loaded into [{table_name}]')

    conn.close()
    print('\n✅ All 9 tables loaded with standardised dates → data/olist.db')


def verify_dates():
    """Quick check that dates now look correct in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT order_purchase_timestamp,
               order_delivered_customer_date,
               order_estimated_delivery_date
        FROM orders
        LIMIT 5
    """, conn)
    conn.close()
    print('\n── Date format check ──')
    print(df.to_string(index=False))


if __name__ == '__main__':
    load_csvs_to_sqlite()
    verify_dates()