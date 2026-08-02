import yaml
import os
import subprocess
import sys
import argparse

def run_experiments(config_file):
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    experiments = data.get('experiments', [])
    print(f"Found {len(experiments)} experiments in {config_file}")
    failed = []
    
    for i, exp in enumerate(experiments):
        name = exp.get('name', f"Experiment {i+1}")
        cfg = exp.get('config')
        desc = exp.get('description', "")
        
        print("\n" + "="*60)
        print(f"Starting Experiment {i+1}/{len(experiments)}: {name}")
        print(f"Description: {desc}")
        print(f"Config File: {cfg}")
        print("="*60 + "\n")
        
        if not os.path.exists(cfg):
            print(f"ERROR: Config file {cfg} not found! Skipping.")
            continue
            
        # commands
        # Assuming we are running from project root
        cmd = [sys.executable, "-u", "main.py", "--config", cfg]
        
        try:
            # Run and stream output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
            if process.returncode != 0:
                print(f"\nExperiment {name} FAILED with exit code {process.returncode}")
                failed.append(name)
            else:
                print(f"\nExperiment {name} COMPLETED SUCCESSFULLY.")
                
        except Exception as e:
            print(f"An error occurred execution experiment {name}: {e}")
            failed.append(name)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS FINISHED.")
    print("Check results_final.csv for summary of best metrics.")
    if failed:
        print(f"Failed experiments: {', '.join(failed)}")
    print("="*60)
    return 1 if failed else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments.yaml', help='Path to master experiment config')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        raise SystemExit(run_experiments(args.config))
    else:
        print(f"Config file {args.config} not found.")
        raise SystemExit(1)
