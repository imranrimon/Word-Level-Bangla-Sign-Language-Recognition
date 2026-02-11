import numpy as np
import pickle
import sys

def inspect_data(data_path, label_path):
    print(f"Inspecting: {data_path}")
    try:
        data = np.load(data_path)
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"Data Shape: {data.shape}") # N, C, T, V, M
    N, C, T, V, M = data.shape
    
    print(f"Label Len: {len(label)}")
    print(f"Unique Labels: {len(set(label))}")
    print(f"Min Label: {min(label)}, Max Label: {max(label)}")
    
    # Stats
    print(f"Min Value: {data.min():.4f}")
    print(f"Max Value: {data.max():.4f}")
    print(f"Mean Value: {data.mean():.4f}")
    
    # Check for "dead" data (all zeros)
    zero_samples = 0
    valid_frames_counts = []
    
    for i in range(len(data)):
        activity = data[i].sum(axis=(0,2,3)) # shape (T,)
        non_zero = np.count_nonzero(activity)
        valid_frames_counts.append(non_zero)
        
        if np.all(data[i] == 0):
            zero_samples += 1

    print(f"Samples with ALL zeros: {zero_samples}")
    print(f"Average Valid Frames: {np.mean(valid_frames_counts):.2f}")
    print(f"Min Valid Frames: {np.min(valid_frames_counts)}")
    print(f"Max Valid Frames: {np.max(valid_frames_counts)}")
    
    import collections
    bins = [0, 1, 5, 10, 30, 60, 120, 300]
    hist = collections.defaultdict(int)
    for v in valid_frames_counts:
        for b in bins:
            if v <= b:
                hist[b] += 1
                break
    print("Valid Frame Distribution (Cumulative <=):")
    for b in bins:
        print(f"  <= {b}: {hist[b]}")

if __name__ == "__main__":
    print("--- TRAIN DATA (IMAGE) ---")
    data_path = 'data/bdsl_img/train_data.npy'
    label_path = 'data/bdsl_img/train_label.pkl'
    inspect_data(data_path, label_path)
