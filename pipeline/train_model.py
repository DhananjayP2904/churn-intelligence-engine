import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'customer_features.csv')
OUT_CSV   = os.path.join(BASE_DIR, 'data', 'customer_scores.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    # ── Load feature table ───────────────────────────────
    print("Loading feature table...")
    df = pd.read_csv(DATA_PATH)
    print(f"   → {len(df):,} customers loaded")

    # ── Encode top_category (text → number) ─────────────
    le = LabelEncoder()
    df['top_category_encoded'] = le.fit_transform(df['top_category'])

    # Save encoder so we can reuse it later
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    # ── Define features and target ───────────────────────
    FEATURES = [
        'frequency',
        'monetary',
        'avg_review_score',
        'min_review_score',
        'late_delivery_pct',
        'late_delivery_count',
        'category_diversity',
        'avg_installments',
        'avg_items_per_order',
        'avg_freight_paid',
        'top_category_encoded',
    ]

    X = df[FEATURES]
    y = df['is_churned']

    print(f"\n── Class distribution ──")
    print(f"   Churned (1) : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"   Active  (0) : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")

    # ── Train / test split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # keeps same churn ratio in both splits
    )
    print(f"\n── Split ──")
    print(f"   Train : {len(X_train):,} rows")
    print(f"   Test  : {len(X_test):,} rows")

    # ── Handle class imbalance ───────────────────────────
    # scale_pos_weight tells XGBoost to penalise missing churned
    # customers more heavily than missing active ones
    ratio = (y == 0).sum() / (y == 1).sum()
    print(f"\n   scale_pos_weight : {ratio:.3f}")

    # ── Train model ──────────────────────────────────────
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = ratio,
        random_state      = 42,
        eval_metric       = 'auc',
        verbosity         = 0,
    )
    model.fit(X_train, y_train)
    print("   → Training complete")

    # ── Evaluate ─────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)
    auc          = roc_auc_score(y_test, y_pred_proba)

    print(f"\n── Model Performance ──")
    print(f"   AUC-ROC : {auc:.4f}   (target > 0.75)")
    print(f"\n── Classification Report ──")
    print(classification_report(y_test, y_pred,
                                target_names=['Active', 'Churned']))

    print(f"\n── Confusion Matrix ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Active  correctly predicted : {cm[0][0]:,}")
    print(f"   Active wrongly predicted churned : {cm[0][1]:,}")
    print(f"   Churned wrongly predicted active : {cm[1][0]:,}")
    print(f"   True Churned correctly predicted : {cm[1][1]:,}")

    # ── Feature importance ───────────────────────────────
    print(f"\n── Feature Importance (top 5) ──")
    importance = pd.DataFrame({
        'feature'   : FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(5).to_string(index=False))

    # ── Save model ───────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, 'churn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved → models/churn_model.pkl")

    # ── Score all customers and save ─────────────────────
    print("\nScoring all customers...")
    df['churn_risk_score'] = (
        model.predict_proba(X[FEATURES])[:, 1] * 100
    ).round(1)

    df['risk_segment'] = pd.cut(
        df['churn_risk_score'],
        bins   = [0, 40, 65, 80, 100],
        labels = ['Low', 'Medium', 'High', 'Critical']
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Scored file saved → data/customer_scores.csv")

    print(f"\n── Risk Segment Distribution ──")
    print(df['risk_segment'].value_counts().to_string())

    return model, auc


if __name__ == '__main__':
    train()