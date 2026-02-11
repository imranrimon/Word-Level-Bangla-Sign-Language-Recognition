
import os
import numpy as np
import sys
import pickle
from tqdm import tqdm

# Add root project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.sign_27 import Graph

def generate_bone_data(data_path, out_path, model_graph='graph.sign_27.Graph'):
    print(f"Generating bone data from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data: (N, C, T, V, M)
    # Using mmap to handle large files if necessary, but we need to write so we load to memory
    try:
        data = np.load(data_path)
    except Exception as e:
        print(f"Failed to load {data_path}: {e}")
        return

    N, C, T, V, M = data.shape
    fp_sp = np.zeros_like(data)

    # Load Graph
    # We need to instantiate the graph to get 'inward' edges (Child -> Parent or vice versa)
    # In sign_27.py: inward = [(child, parent), ...] actually let's verify direction
    # inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
    # inward_ori_index = [(5, 6), (5, 7)...] where 5 is connected to 6.
    # We typically want Bone = Target - Source.
    
    # Let's check a known structure.
    # WLASL/Nose (0) is usually root.
    # In sign_27.py: (5,6) -> (0,1). 0 is likely nose, 1 is shoulder.
    # If 0 is root, 1 is child.
    # The pairs in 'self.inward' are defined as edges.
    
    # We will compute Bone = Node_v1 - Node_v2
    # For each pair (v1, v2), we treat v1 as child, v2 as parent?
    # Or simply iterate all edges. 
    # Standard implementation:
    # For the root node, bone is 0.
    # For other nodes, bone is vector from parent.
    
    graph = Graph(labeling_mode='spatial')
    # inward format: List of (u, v).
    # In CTR-GCN and others:
    # for v1, v2 in graph.inward:
    #     fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
    
    # We need to know which is child and which is parent to assign correctly.
    # But actually, 'inward' usually defines the tree structure directed towards center?
    # No, 'inward' usually means (node, neighbor).
    
    # Let's just use the 'inward' list which defines the connections.
    # We simply iterate the edges.
    
    # Note: sign_27.py defines:
    # inward = [(i-5, j-5)...]
    # (5,6) -> (0,1).
    # If 0 (Nose) is root. 1 (Shoulder) is child.
    # Vector should represent the limb.
    # Usually we assign the bone vector to the CHILD node index.
    # So if (0,1) is an edge, and 1 is child, then Bone[1] = Joint[1] - Joint[0].
    # Bone[0] (root) = 0.
    
    # Let's assume the graph definition in sign_27.py implies v1 is closer to root?
    # (5,6) -> (0,1). (5,7) -> (0,2).
    # If 0 is Nose, 1 is R-Shoulder.
    # It seems index 0 is center.
    # So v1=0, v2=1.
    # Edge is (0,1).
    # We want Bone[1] = Joint[1] - Joint[0].
    # So: fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]
    
    for v1, v2 in graph.inward:
        # v1 is index 0 (source/center), v2 is index 1 (target/distal)
        # Verify direction: (5,6) -> (0,1).
        # v1=0, v2=1.
        # We assign vector to v2.
        fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]
        
    print(f"Bone data generated. Shape: {fp_sp.shape}")
    np.save(out_path, fp_sp)
    print(f"Saved to {out_path}")

def copy_label(label_path, out_path):
    if not os.path.exists(label_path):
        return
    import shutil
    shutil.copy2(label_path, out_path)
    print(f"Copied label to {out_path}")

if __name__ == "__main__":
    # Video Data
    generate_bone_data('data/bdsl/train_data.npy', 'data/bdsl/train_data_bone.npy')
    generate_bone_data('data/bdsl/val_data.npy', 'data/bdsl/val_data_bone.npy')
    # Copy labels just for consistency in naming if we wanted, but we can reuse original labels in config.
    # But having matching filenames is nice.
    # Actually config asks for label_path. We can point to original.
    
    # Image Data
    generate_bone_data('data/bdsl_img/train_data.npy', 'data/bdsl_img/train_data_bone.npy')
    generate_bone_data('data/bdsl_img/val_data.npy', 'data/bdsl_img/val_data_bone.npy')
