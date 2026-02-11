
import os
import argparse
import pickle
import numpy as np

def fuse_scores(work_dir_joint, work_dir_bone, alpha=1.0):
    print(f"Fusing scores from:")
    print(f"  Joint: {work_dir_joint}")
    print(f"  Bone:  {work_dir_bone}")
    
    # helper to find best pkl
    def get_best_pkl(work_dir):
        eval_dir = os.path.join(work_dir, "eval_results")
        if not os.path.exists(eval_dir):
            return None
        import glob
        pkls = glob.glob(os.path.join(eval_dir, "*.pkl"))
        best_pkl = None
        for p in pkls:
            if "best_acc.pkl" in p:
                return p
        # if no best_acc, find one with highest accuracy in name "epoch_E_ACC.pkl"
        best_acc = -1
        for p in pkls:
            try:
                # epoch_50_0.85.pkl
                parts = os.path.basename(p).replace('.pkl','').split('_')
                acc = float(parts[-1])
                if acc > best_acc:
                    best_acc = acc
                    best_pkl = p
            except:
                pass
        return best_pkl

    joint_pkl = get_best_pkl(work_dir_joint)
    bone_pkl = get_best_pkl(work_dir_bone)
    
    if not joint_pkl or not bone_pkl:
        print("Could not find result .pkl files in one or both directories.")
        return

    print(f"Loading Joint: {joint_pkl}")
    with open(joint_pkl, 'rb') as f:
        r1 = pickle.load(f)
        
    print(f"Loading Bone: {bone_pkl}")
    with open(bone_pkl, 'rb') as f:
        r2 = pickle.load(f)

    # Intersection of keys
    right_num = 0
    total_num = 0
    
    # Load labels to calculate accuracy
    # infer from work_dir_joint config
    # Hack: assume standard path
    dataset_name = os.path.basename(work_dir_joint).replace('_graph','').replace('_bone','')
    label_path = f"data/{dataset_name}/val_label.pkl"
    if not os.path.exists(label_path):
        # try bdsl
        label_path = "data/bdsl/val_label.pkl"
        
    print(f"Loading labels from: {label_path}")
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f)
    
    match_dict = dict(zip(sample_name, label))
    
    fused_score_dict = {}

    for k in r1.keys():
        if k in r2:
            s1 = r1[k]
            s2 = r2[k]
            
            # FUSION
            score = s1 + alpha * s2
            
            fused_score_dict[k] = score
            
            pred = np.argmax(score)
            if k in match_dict:
                gt = match_dict[k]
                total_num += 1
                if pred == gt:
                    right_num += 1
    
    acc = right_num / total_num
    print("--------------------------------------------------")
    print(f"Joint + Bone Fusion Accuracy: {acc*100:.2f}%")
    print("--------------------------------------------------")
    
    # Visualization?
    # We can save this fused dict as a pseudo-result and run visualize_results on it?
    # Or just return.
    
    out_file = "fused_results.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(fused_score_dict, f)
    print(f"Saved fused scores to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_dir', required=True)
    parser.add_argument('--bone_dir', required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()
    
    fuse_scores(args.joint_dir, args.bone_dir, args.alpha)
