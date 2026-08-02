
import pandas as pd
import argparse
import os

def summarize_all(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        # Load without stripping to see raw strings if needed
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty.")
        return

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    
    # Identify accuracy column
    acc_col = 'Top1_Acc' if 'Top1_Acc' in df.columns else 'Accuracy'
    if acc_col not in df.columns:
        print(f"No accuracy column found. Columns: {df.columns.tolist()}")
        return

    # Basic cleaning
    df['Experiment'] = df['Experiment'].astype(str).str.strip()
    if 'WorkDir' in df.columns:
        df['WorkDir'] = df['WorkDir'].astype(str).str.strip()

    print(f"Total entries: {len(df)}")
    
    # Get best results group by Experiment and WorkDir
    # We use WorkDir to distinguish separate runs of the same model
    unique_runs = df[['Experiment', 'WorkDir']].drop_duplicates()
    
    print(f"\n{'='*100}")
    print(f"{'EXPERIMENT RESULTS SUMMARY':^100}")
    print(f"{'='*100}")
    print("Top5@Top1 is the Top-5 value from the same epoch as Best Top1.")
    print(f"{'Experiment':<25} | {'WorkDir':<32} | {'Best Top1':<10} | {'Top5@Top1':<10} | {'Best Top5':<10} | {'Epoch@Top1':<10} | {'Epoch@Top5':<10}")
    print(f"{'-'*130}")

    for _, row in unique_runs.iterrows():
        exp = row['Experiment']
        wd = row['WorkDir']
        
        run_data = df[(df['Experiment'] == exp) & (df['WorkDir'] == wd)]
        
        if run_data.empty: continue
        
        run_data = run_data.copy()
        run_data[acc_col] = pd.to_numeric(run_data[acc_col], errors='coerce')
        if 'Top5_Acc' in run_data.columns:
            run_data['Top5_Acc'] = pd.to_numeric(run_data['Top5_Acc'], errors='coerce')

        best_idx = run_data[acc_col].idxmax()
        best_row = run_data.loc[best_idx]
        
        top1 = f"{best_row[acc_col]:.4f}"
        top5_at_top1 = "N/A"
        best_top5 = "N/A"
        epoch_top5 = "N/A"
        if 'Top5_Acc' in df.columns and pd.notna(best_row['Top5_Acc']):
            top5_at_top1 = f"{best_row['Top5_Acc']:.4f}"
            best_top5_idx = run_data['Top5_Acc'].idxmax()
            best_top5_row = run_data.loc[best_top5_idx]
            best_top5 = f"{best_top5_row['Top5_Acc']:.4f}"
            epoch_top5 = best_top5_row['Epoch']
            
        epoch = best_row['Epoch']
        
        # Shorten WorkDir for display
        wd_display = os.path.basename(wd.rstrip('/'))
        if not wd_display: wd_display = wd
        
        print(f"{exp:<25} | {wd_display:<32} | {top1:<10} | {top5_at_top1:<10} | {best_top5:<10} | {str(epoch):<10} | {str(epoch_top5):<10}")

    print(f"{'='*130}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results_final.csv', help='CSV file')
    args = parser.parse_args()
    summarize_all(args.csv)
