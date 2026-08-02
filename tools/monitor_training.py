"""
Real-time training monitor for results_final.csv.
"""
import argparse
import os
import time
from datetime import datetime

import pandas as pd


def build_summary(df):
    df = df.copy()
    df["Top1_Acc"] = pd.to_numeric(df["Top1_Acc"], errors="coerce")
    df["Top5_Acc"] = pd.to_numeric(df["Top5_Acc"], errors="coerce")

    rows = []
    for exp, exp_df in df.groupby("Experiment"):
        best_top1_row = exp_df.loc[exp_df["Top1_Acc"].idxmax()]
        best_top5_row = exp_df.loc[exp_df["Top5_Acc"].idxmax()]
        rows.append({
            "Experiment": exp,
            "Best_Top1": best_top1_row["Top1_Acc"],
            "Top5_at_Best_Top1": best_top1_row["Top5_Acc"],
            "Best_Top5": best_top5_row["Top5_Acc"],
            "Epoch_at_Best_Top1": best_top1_row["Epoch"],
            "Epoch_at_Best_Top5": best_top5_row["Epoch"],
        })
    return pd.DataFrame(rows).sort_values("Best_Top1", ascending=False)


def monitor_training(csv_path="results_final.csv", refresh_interval=10):
    """Monitor training progress in real time."""
    print("=" * 80)
    print("TRAINING MONITOR - Press Ctrl+C to exit")
    print("=" * 80)

    last_size = 0
    last_update = None

    try:
        while True:
            if not os.path.exists(csv_path):
                print(f"Waiting for {csv_path} to be created...")
                time.sleep(refresh_interval)
                continue

            current_size = os.path.getsize(csv_path)
            if current_size != last_size:
                last_size = current_size
                last_update = datetime.now()

                df = pd.read_csv(csv_path)
                os.system("cls" if os.name == "nt" else "clear")

                print("=" * 80)
                print(f"TRAINING MONITOR - Last Update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                print()

                print("BEST RESULTS BY EXPERIMENT:")
                print("Top5_at_Best_Top1 is from the same epoch as Best_Top1.")
                print("-" * 80)
                print(build_summary(df).to_string(index=False))
                print()

                print("LATEST 5 TRAINING ENTRIES:")
                print("-" * 80)
                latest_columns = ["Timestamp", "Experiment", "Epoch", "Top1_Acc", "Top5_Acc"]
                print(df.tail(5)[latest_columns].to_string(index=False))
                print()

                if len(df) > 0:
                    latest_exp = df.iloc[-1]["Experiment"]
                    latest_epoch = df.iloc[-1]["Epoch"]
                    exp_df = df[df["Experiment"] == latest_exp].copy()
                    exp_df["Top1_Acc"] = pd.to_numeric(exp_df["Top1_Acc"], errors="coerce")
                    exp_df["Top5_Acc"] = pd.to_numeric(exp_df["Top5_Acc"], errors="coerce")
                    top1_row = exp_df.loc[exp_df["Top1_Acc"].idxmax()]

                    print(f"CURRENT EXPERIMENT: {latest_exp}")
                    print(f"   Epoch: {latest_epoch}")
                    print(f"   Best Top-1: {exp_df['Top1_Acc'].max():.4f}")
                    print(f"   Top-5 at Best Top-1: {top1_row['Top5_Acc']:.4f}")
                    print(f"   Best Top-5 independently: {exp_df['Top5_Acc'].max():.4f}")
                    print(f"   Total Entries: {len(exp_df)}")
                print()

                print("=" * 80)
                print(f"Refreshing every {refresh_interval} seconds... (Ctrl+C to exit)")
                print("=" * 80)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--csv", default="results_final.csv", help="Path to results CSV")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    args = parser.parse_args()

    monitor_training(args.csv, args.interval)
