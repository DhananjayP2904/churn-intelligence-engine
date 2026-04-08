import os
import time
import pandas as pd
from groq import Groq

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IN_PATH  = os.path.join(BASE_DIR, 'data', 'customer_scores.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'ai_churn_briefs.csv')

# ── Paste your Groq key here ──────────────────────────
GROQ_API_KEY = "use_your_own_api"


def generate_brief(client, row):
    prompt = f"""You are a senior CRM analyst at a large e-commerce company.
Write a churn brief for the retention team about this customer.

Customer data:
- Total orders placed      : {row['frequency']}
- Lifetime spend           : ${row['monetary']:.2f}
- Average review score     : {row['avg_review_score']:.1f} / 5
- Lowest review ever given : {row['min_review_score']:.0f} / 5
- Late delivery rate       : {row['late_delivery_pct']:.0f}% of their orders arrived late
- Number of late deliveries: {row['late_delivery_count']:.0f}
- Product categories bought: {row['category_diversity']:.0f} different categories
- Top category             : {row['top_category']}
- Avg payment installments : {row['avg_installments']:.1f}
- Churn risk score         : {row['churn_risk_score']:.0f} / 100
- Risk segment             : {row['risk_segment']}

Write exactly 3 sentences, plain paragraph, no bullet points, no headers:
Sentence 1 - WHY AT RISK: Specific reason based on their data, not generic.
Sentence 2 - RECOMMENDED ACTION: One precise retention action for this week.
Sentence 3 - REVENUE IMPACT: Cost of inaction in dollar terms.

Maximum 80 words total."""

    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",  # free, fast, high quality
        messages = [{"role": "user", "content": prompt}],
        max_tokens      = 150,
        temperature     = 0.7,
    )
    return response.choices[0].message.content.strip()


def run_ai_layer(max_customers=50):
    print("Loading scored customers...")
    df = pd.read_csv(IN_PATH)

    high_risk = df[df['risk_segment'].isin(['Critical', 'High'])].copy()
    high_risk  = high_risk.sort_values('churn_risk_score', ascending=False)
    high_risk  = high_risk.head(max_customers).reset_index(drop=True)

    print(f"   → {len(df):,} total customers")
    print(f"   → Generating briefs for top {max_customers} by risk score\n")

    client = Groq(api_key=GROQ_API_KEY)

    briefs = []
    for i, row in high_risk.iterrows():
        print(f"   [{i+1}/{len(high_risk)}] "
              f"Customer {row['customer_unique_id'][:8]}...  "
              f"risk={row['churn_risk_score']:.0f}  "
              f"segment={row['risk_segment']}")

        try:
            brief = generate_brief(client, row)
            briefs.append({
                'customer_unique_id' : row['customer_unique_id'],
                'churn_risk_score'   : row['churn_risk_score'],
                'risk_segment'       : row['risk_segment'],
                'frequency'          : row['frequency'],
                'monetary'           : row['monetary'],
                'avg_review_score'   : row['avg_review_score'],
                'late_delivery_pct'  : row['late_delivery_pct'],
                'top_category'       : row['top_category'],
                'ai_brief'           : brief,
            })
            time.sleep(0.5)  # stay within free rate limits

        except Exception as e:
            print(f"      ⚠ Error: {e}")
            briefs.append({
                'customer_unique_id' : row['customer_unique_id'],
                'churn_risk_score'   : row['churn_risk_score'],
                'risk_segment'       : row['risk_segment'],
                'frequency'          : row['frequency'],
                'monetary'           : row['monetary'],
                'avg_review_score'   : row['avg_review_score'],
                'late_delivery_pct'  : row['late_delivery_pct'],
                'top_category'       : row['top_category'],
                'ai_brief'           : 'Brief unavailable',
            })

    briefs_df = pd.DataFrame(briefs)
    briefs_df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Briefs saved → data/ai_churn_briefs.csv")
    print(f"   Total generated : {len(briefs_df):,}")
    print(f"\n── Sample briefs ──")
    for _, row in briefs_df.head(3).iterrows():
        print(f"\nCustomer : {row['customer_unique_id'][:16]}...")
        print(f"Risk     : {row['churn_risk_score']:.0f}/100  |  "
              f"Spend: ${row['monetary']:.0f}  |  "
              f"Category: {row['top_category']}")
        print(f"Brief    : {row['ai_brief']}")
        print("─" * 60)

    return briefs_df


if __name__ == '__main__':
    run_ai_layer(max_customers=50)