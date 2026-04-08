import schedule
import time
import sys
import os
from datetime import datetime

# Add root to path so we can import from pipeline/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline.load_data   import load_csvs_to_sqlite
from pipeline.build_features import build_features
from pipeline.train_model import train
from pipeline.ai_layer    import run_ai_layer


def run_full_pipeline():
    start = datetime.now()
    print(f"\n{'='*55}")
    print(f"  PIPELINE STARTED — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    try:
        print("[ 1/4 ] Loading & refreshing data...")
        load_csvs_to_sqlite()

        print("\n[ 2/4 ] Building feature table...")
        build_features()

        print("\n[ 3/4 ] Training churn model...")
        train()

        print("\n[ 4/4 ] Generating AI briefs...")
        run_ai_layer(max_customers=50)

        end = datetime.now()
        duration = (end - start).seconds
        print(f"\n{'='*55}")
        print(f"  ✅ PIPELINE COMPLETE — {duration}s elapsed")
        print(f"  Dashboard data is ready for refresh in Power BI")
        print(f"{'='*55}\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")


if __name__ == '__main__':
    # Run immediately once
    run_full_pipeline()

    # Then schedule every Sunday at midnight
    schedule.every().sunday.at("00:00").do(run_full_pipeline)
    print("⏰ Scheduler running — pipeline will auto-run every Sunday midnight")
    print("   Press Ctrl+C to stop\n")

    while True:
        schedule.run_pending()
        time.sleep(60)