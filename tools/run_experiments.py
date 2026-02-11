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
                # Optional: break or continue? User likely wants to run all.
                # continue 
            else:
                print(f"\nExperiment {name} COMPLETED SUCCESSFULLY.")
                
        except Exception as e:
            print(f"An error occurred execution experiment {name}: {e}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS FINISHED.")
    print("Check results.csv for summary of best metrics.")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments.yaml', help='Path to master experiment config')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        run_experiments(args.config)
    else:
        print(f"Config file {args.config} not found.")
