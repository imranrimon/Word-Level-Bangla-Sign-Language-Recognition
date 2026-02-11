"""
Real-Time Training Monitor
Monitors results_final.csv and displays live training progress
"""
import pandas as pd
import time
import os
from datetime import datetime

def monitor_training(csv_path='results_final.csv', refresh_interval=10):
    """Monitor training progress in real-time"""
    
    print("="*80)
    print("🔍 TRAINING MONITOR - Press Ctrl+C to exit")
    print("="*80)
    
    last_size = 0
    last_update = None
    
    try:
        while True:
            if not os.path.exists(csv_path):
                print(f"Waiting for {csv_path} to be created...")
                time.sleep(refresh_interval)
                continue
            
            # Check if file has been updated
            current_size = os.path.getsize(csv_path)
            
            if current_size != last_size:
                last_size = current_size
                last_update = datetime.now()
                
                # Read and display latest results
                df = pd.read_csv(csv_path)
                
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("="*80)
                print(f"🔍 TRAINING MONITOR - Last Update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                print()
                
                # Summary by experiment
                print("📊 BEST RESULTS BY EXPERIMENT:")
                print("-"*80)
                summary = df.groupby('Experiment').agg({
                    'Top1_Acc': 'max',
                    'Top5_Acc': 'max',
                    'Epoch': 'max'
                }).reset_index()
                summary = summary.sort_values('Top1_Acc', ascending=False)
                print(summary.to_string(index=False))
                print()
                
                # Latest 5 entries
                print("📝 LATEST 5 TRAINING ENTRIES:")
                print("-"*80)
                latest = df.tail(5)[['Timestamp', 'Experiment', 'Epoch', 'Top1_Acc', 'Top5_Acc']]
                print(latest.to_string(index=False))
                print()
                
                # Current experiment status
                if len(df) > 0:
                    latest_exp = df.iloc[-1]['Experiment']
                    latest_epoch = df.iloc[-1]['Epoch']
                    exp_df = df[df['Experiment'] == latest_exp]
                    
                    print(f"🔥 CURRENT EXPERIMENT: {latest_exp}")
                    print(f"   Epoch: {latest_epoch}")
                    print(f"   Best Top-1: {exp_df['Top1_Acc'].max():.4f}")
                    print(f"   Best Top-5: {exp_df['Top5_Acc'].max():.4f}")
                    print(f"   Total Entries: {len(exp_df)}")
                print()
                
                print("="*80)
                print(f"Refreshing every {refresh_interval} seconds... (Ctrl+C to exit)")
                print("="*80)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--csv', default='results_final.csv', help='Path to results CSV')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')
    args = parser.parse_args()
    
    monitor_training(args.csv, args.interval)
