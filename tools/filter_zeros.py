import numpy as np
import pickle
import sys
import os

def filter_data(data_path, label_path, suffix='_filtered'):
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f)
    
    label = np.array(label)
    sample_name = np.array(sample_name)
    
    print(f"Original Shape: {data.shape}")
    
    # Identify valid indices
    # Sum over C, T, V, M
    energy = np.abs(data).sum(axis=(1, 2, 3, 4))
    valid_indices = np.where(energy > 0)[0]
    
    print(f"Valid Samples: {len(valid_indices)} / {len(data)}")
    print(f"Removed: {len(data) - len(valid_indices)}")
    
    # Filter
    new_data = data[valid_indices]
    new_label = label[valid_indices]
    new_sample_name = sample_name[valid_indices]
    
    base, ext = os.path.splitext(data_path)
    new_data_path = f"{base}{suffix}{ext}"
    
    base_lbl, ext_lbl = os.path.splitext(label_path)
    new_label_path = f"{base_lbl}{suffix}{ext_lbl}"
    
    print(f"Saving to {new_data_path}...")
    np.save(new_data_path, new_data)
    
    print(f"Saving to {new_label_path}...")
    with open(new_label_path, 'wb') as f:
        pickle.dump((new_sample_name.tolist(), new_label.tolist()), f)
        
    print("Done.")

if __name__ == "__main__":
    # Joint Image Data
    print("--- Filtering Joint Image ---")
    filter_data('data/bdsl_img/train_data.npy', 'data/bdsl_img/train_label.pkl')
    filter_data('data/bdsl_img/val_data.npy', 'data/bdsl_img/val_label.pkl')
    
    # Bone Image Data (if exists)
    if os.path.exists('data/bdsl_img/train_data_bone.npy'):
        print("\n--- Filtering Bone Image ---")
        filter_data('data/bdsl_img/train_data_bone.npy', 'data/bdsl_img/train_label.pkl')
        filter_data('data/bdsl_img/val_data_bone.npy', 'data/bdsl_img/val_label.pkl')
