import pandas as pd
import sqlite3

import os
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'olist.db')

def run_query(sql, label):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn)
    conn.close()
    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':

    # 1. How many unique customers do we have?
    run_query("""
        SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers
        FROM customers
    """, "Total unique customers")

    # 2. How many orders and what statuses exist?
    run_query("""
        SELECT order_status,
               COUNT(*) AS total_orders
        FROM orders
        GROUP BY order_status
        ORDER BY total_orders DESC
    """, "Orders by status")

    # 3. What are the top 10 product categories by sales?
    run_query("""
        SELECT ct.product_category_name_english AS category,
               COUNT(oi.order_id)               AS total_orders,
               ROUND(SUM(oi.price), 2)          AS total_revenue
        FROM order_items oi
        JOIN products p
          ON oi.product_id = p.product_id
        JOIN category_translation ct
          ON p.product_category_name = ct.product_category_name
        GROUP BY category
        ORDER BY total_revenue DESC
        LIMIT 10
    """, "Top 10 categories by revenue")

    # 4. What is the overall average review score?
    run_query("""
        SELECT ROUND(AVG(review_score), 2)  AS avg_review_score,
               COUNT(*)                     AS total_reviews
        FROM reviews
    """, "Average review score")

    # 5. How many customers made more than 1 purchase (repeat buyers)?
    run_query("""
        SELECT
            SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) AS repeat_customers,
            SUM(CASE WHEN order_count = 1 THEN 1 ELSE 0 END) AS one_time_customers,
            COUNT(*) AS total_customers
        FROM (
            SELECT c.customer_unique_id,
                   COUNT(o.order_id) AS order_count
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_status = 'delivered'
            GROUP BY c.customer_unique_id
        )
    """, "Repeat vs one-time customers")

    # 6. Late delivery rate
    run_query("""
        SELECT
            COUNT(*) AS total_delivered,
            SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date
                     THEN 1 ELSE 0 END) AS late_deliveries,
            ROUND(
                100.0 * SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date
                                 THEN 1 ELSE 0 END) / COUNT(*), 2
            ) AS late_delivery_pct
        FROM orders
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
    """, "Late delivery rate")