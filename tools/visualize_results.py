
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob

def parse_log(log_path):
    print(f"Parsing log: {log_path}")
    if not os.path.exists(log_path):
        print("Log file not found.")
        return None, None

    train_loss = []
    val_loss = []
    val_acc = []
    epochs = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_epoch = 0
    for line in lines:
        # Training epoch: 1
        if "Training epoch:" in line:
            current_epoch = int(line.split(":")[-1].strip())
            epochs.append(current_epoch)
        
        # Batch(50/465) done. Loss: 1.2345
        # This is batch loss, maybe too noisy. 
        # But wait, main.py logs 'train_loss' to wandb, but prints batch loss.
        # We can try to average batch losses or see if there is an epoch loss print?
        # Main.py validation: "Mean test loss of X batches: ..."
        
        # main.py line 430: timer stats... then evaluation.
        # It doesn't seem to print "Mean Training Loss".
        # So we might have to scrape "Batch ... Loss: ..." and average them for the epoch.
        pass

    # Regex for batch loss
    # \tBatch(697/697) done. Loss: 0.6974  lr:0.000100
    batch_losses = {} # epoch -> list of losses
    
    current_epoch_scan = 0
    for line in lines:
        if "Training epoch:" in line:
            current_epoch_scan = int(line.split(":")[-1].strip())
            if current_epoch_scan not in batch_losses:
                batch_losses[current_epoch_scan] = []
        
        if "Batch(" in line and "Loss:" in line:
            parts = line.split("Loss:")
            try:
                loss_val = float(parts[1].split()[0])
                if current_epoch_scan > 0:
                    batch_losses[current_epoch_scan].append(loss_val)
            except:
                pass
                
        # Eval Accuracy:  0.123
        if "Eval Accuracy:" in line:
             # Match to latest epoch
             try:
                 acc = float(line.split("Eval Accuracy:")[1].split()[0])
                 val_acc.append((current_epoch_scan, acc))
             except:
                 pass

    # Aggregate Training Loss
    epochs_sorted = sorted(batch_losses.keys())
    t_loss = []
    t_epochs = []
    for ep in epochs_sorted:
        if len(batch_losses[ep]) > 0:
            avg_loss = sum(batch_losses[ep]) / len(batch_losses[ep])
            t_loss.append(avg_loss)
            t_epochs.append(ep)

    # Validation Loss? "Mean test loss of 115 batches: 1.234"
    v_loss_dict = {}
    current_epoch_scan = 0
    for line in lines:
        if "Training epoch:" in line:
            current_epoch_scan = int(line.split(":")[-1].strip())
            # Eval happens after training epoch
        if "Mean test loss of" in line:
             try:
                 vl = float(line.strip().split(":")[-1].replace('.', '', 1)) # handled by float usually but check trailing dot
                 # actually "1.234." -> float("1.234.") works? No.
                 # "batches: 1.6372."
                 val_str = line.strip().split(":")[-1].strip()
                 if val_str.endswith('.'):
                     val_str = val_str[:-1]
                 v_loss_dict[current_epoch_scan] = float(val_str)
             except:
                 pass
                 
    v_loss = []
    v_epochs = []
    for ep in sorted(v_loss_dict.keys()):
        v_loss.append(v_loss_dict[ep])
        v_epochs.append(ep)

    return t_epochs, t_loss, v_epochs, v_loss, val_acc

def plot_trajectories(t_epochs, t_loss, v_epochs, v_loss, val_acc, output_dir):
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(t_epochs, t_loss, label='Training Loss')
    if v_loss:
        plt.plot(v_epochs, v_loss, label='Validation Loss')
    # Use validation epochs for x-axis if available, else training
    plt.title('Training and Validation Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    acc_epochs = [x[0] for x in val_acc]
    acc_values = [x[1] * 100 for x in val_acc] # Convert to percentage? val_acc is usually 0.5? Check log. 
    # Log says: "Top1: 12.34%" -> printed. But "Eval Accuracy:  0.1234" usually.
    # main.py: "Eval Accuracy: ", accuracy (which is top_k result)
    # accuracy > best_acc
    # Let's assume it's 0-1 if usage is top_k(score, 1).
    
    plt.plot(acc_epochs, [x * 100 for x in [y[1] for y in val_acc]], label='Top-1 Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, 'learning_trajectory.png')
    plt.savefig(save_path)
    print(f"Saved trajectory plot to {save_path}")
    plt.close()

def plot_confusion_matrix(work_dir, output_dir):
    # Load evaluation results from .pkl files
    # main.py saves: work_dir/Experiment/eval_results/epoch_X_acc.pkl -> score_dict
    # score_dict is {sample_name: score}
    # We need labels provided by dataset.config or pickle.
    
    # We also need true labels.
    # config.yaml -> test_feeder_args -> label_path
    
    # Find config
    config_path = os.path.join(work_dir, "config.yaml") # Often outputted? 
    # Or checking log.txt parameters line
    
    # Assuming standard structure
    # Try to find eval_results
    eval_dir = os.path.join(work_dir, "eval_results")
    if not os.path.exists(eval_dir):
        print(f"No eval_results found in {work_dir}")
        return

    # Get best accuracy pkl
    pkls = glob.glob(os.path.join(eval_dir, "*.pkl"))
    if not pkls:
        print("No .pkl result files found.")
        return
        
    # Sort by accuracy (usually in filename) or modification time?
    # Filename format: epoch_E_ACC.pkl or best_acc.pkl
    best_pkl = None
    for p in pkls:
        if "best_acc.pkl" in p:
            best_pkl = p
            break
    
    if not best_pkl:
        best_pkl = pkls[-1] # take last one
        
    print(f"Using {best_pkl} for Confusion Matrix")
    
    with open(best_pkl, 'rb') as f:
        score_dict = pickle.load(f)
        
    # score_dict: {filename: score_vector}
    
    # We need True Labels.
    # Load val_label.pkl
    # Try to infer path. bdsl_img -> data/bdsl_img/val_label.pkl
    # Let's look for val_label.pkl relative to work_dir? No.
    # Hardcode logical guess based on directory name
    
    dataset_name = os.path.basename(work_dir) # bdsl_img or bdsl_graph
    label_path = f"data/{dataset_name}/val_label.pkl"
    
    if not os.path.exists(label_path):
        # check parent
        parent_label = f"../data/{dataset_name}/val_label.pkl"
        if os.path.exists(parent_label):
            label_path = parent_label
        else:
             print(f"Could not find label file at {label_path}")
             return

    print(f"Loading labels from {label_path}")
    with open(label_path, 'rb') as f:
        sample_name, labels = pickle.load(f)
    
    # Map labels
    true_labels = []
    pred_labels = []
    
    for i, name in enumerate(sample_name):
        if name in score_dict:
            score = score_dict[name]
            pred = np.argmax(score)
            true_labels.append(labels[i])
            pred_labels.append(pred)
            
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True, help='Path to work_dir/experiment')
    args = parser.parse_args()
    
    if not os.path.exists(args.work_dir):
        print(f"Work dir {args.work_dir} does not exist.")
        exit(1)
        
    log_file = os.path.join(args.work_dir, "log.txt")
    output_dir = args.work_dir
    
    t_epochs, t_loss, v_epochs, v_loss, val_acc = parse_log(log_file)
    if t_epochs:
        plot_trajectories(t_epochs, t_loss, v_epochs, v_loss, val_acc, output_dir)
    else:
        print("No training data found in log.")
        
    try:
        plot_confusion_matrix(args.work_dir, output_dir)
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
