import numpy as np

def check_variance(data_path):
    print(f"Loading {data_path}...")
    try:
        data = np.load(data_path)
    except:
        print("File not found.")
        return

    # data: N, C, T, V, M
    # Calculate variance along T (axis 2)
    # We want average variance per sample, per joint.
    
    print(f"Shape: {data.shape}")
    
    # Take a subset to save time
    subset = data[:100]
    
    # Var along T
    vars = np.var(subset, axis=2) # N, C, V, M
    avg_var = np.mean(vars)
    max_var = np.max(vars)
    
    print(f"Average Temporal Variance: {avg_var:.6f}")
    print(f"Max Temporal Variance: {max_var:.6f}")
    
    if avg_var < 0.0001:
        print("CONCLUSION: STATIC IMAGES (Effective T=1)")
    else:
        print("CONCLUSION: VIDEO SEQUENCE (Effective T>1)")

if __name__ == "__main__":
    check_variance('data/bdsl_img/train_data_filtered.npy')
