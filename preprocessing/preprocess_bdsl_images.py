
import os
import cv2
import sys
import glob
import numpy as np
import pickle
import mediapipe as mp
import argparse
from tqdm import tqdm

# Create a module path for loading 'feeders' from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feeders.tools import openpose_matchrse

# Define Keypoint Mapping
# Body (0-6)
BODY_INDICES = [0, 12, 11, 14, 13, 16, 15]
# Hand
HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

def process_image(args):
    file_path, label, label_name = args
    
    mp_holistic = mp.solutions.holistic
    try:
        # Static image mode is True for images
        holistic = mp_holistic.Holistic(
            static_image_mode=True, 
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"Error init holistic: {e}")
        return None

    image = cv2.imread(file_path)
    if image is None:
        print(f"Error reading image: {file_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Shape: (C, V) aka (3, 27)
    frame_kps = np.zeros((3, 27)) # x, y, confidence
    
    def get_kp(landmark_list, idx):
        if landmark_list and idx < len(landmark_list.landmark):
            lm = landmark_list.landmark[idx]
            return [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
        return [0, 0, 0]

    # Body
    if results.pose_landmarks:
        for i, idx in enumerate(BODY_INDICES):
            frame_kps[:, i] = get_kp(results.pose_landmarks, idx)
    
    # Right Hand (7-16)
    if results.right_hand_landmarks:
        for i, idx in enumerate(HAND_INDICES):
            frame_kps[:, 7 + i] = get_kp(results.right_hand_landmarks, idx)

    # Left Hand (17-26)
    if results.left_hand_landmarks:
        for i, idx in enumerate(HAND_INDICES):
            frame_kps[:, 17 + i] = get_kp(results.left_hand_landmarks, idx)
            
    holistic.close()
    
    # Expand to temporal dimension (T=120) "Frozen Video"
    # (3, 27) -> (1, 3, 27) -> (120, 3, 27) -> (3, 120, 27)
    
    T = 120
    # repeat
    data = np.tile(frame_kps[np.newaxis, :, :], (T, 1, 1)) # (T, C, V)
    
    # (T, C, V) -> (C, T, V)
    data = data.transpose(1, 0, 2) 
    
    # Add Person Dimension (M=1) -> (C, T, V, M)
    data = data[..., np.newaxis] # (C, T, V, M)
    
    return (data, label, label_name, file_path)

def preprocess(dataset_root, output_dir):
    print("Starting preprocess function for IMAGES")
    sys.stdout.flush()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Listing directory: {dataset_root}")
    sys.stdout.flush()
    classes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    sys.stdout.flush()
    
    all_tasks = []
    
    for cls_name in classes:
        cls_dir = os.path.join(dataset_root, cls_name)
        print(f"Scanning class {cls_name}...", end='\r')
        sys.stdout.flush()
        # Look for jpg, png, jpeg
        img_files = glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(cls_dir, "*.JPG"))
                    
        for img_file in img_files:
            all_tasks.append((img_file, class_to_idx[cls_name], cls_name))
            
    print(f"\nFound total {len(all_tasks)} images.")
    sys.stdout.flush()

    results = []
    for i, task in enumerate(all_tasks):
        if i % 100 == 0:
            print(f"Processing {i}/{len(all_tasks)}: {task[0]}")
            sys.stdout.flush()
        try:
            res = process_image(task)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"Error processing {task[0]}: {e}")
            sys.stdout.flush()
    
    print(f"Successfully processed {len(results)} images")
    sys.stdout.flush()
    
    # Split Train/Val (80/20)
    import random
    random.seed(42)
    random.shuffle(results)
    
    split_idx = int(0.8 * len(results))
    train_data_list = results[:split_idx]
    val_data_list = results[split_idx:]
    
    def save_split(data_list, split_name):
        if not data_list:
            return
            
        N = len(data_list)
        C = 3
        T = 120
        V = 27
        M = 1
        
        # Pre-allocate
        large_data = np.zeros((N, C, T, V, M), dtype=np.float32)
        labels = []
        sample_names = []
        
        for i, (d, label, label_name, vid_path) in enumerate(data_list):
            large_data[i] = d
            labels.append(label)
            sample_names.append(os.path.basename(vid_path))
            
        np.save(os.path.join(output_dir, f'{split_name}_data.npy'), large_data)
        
        with open(os.path.join(output_dir, f'{split_name}_label.pkl'), 'wb') as f:
            pickle.dump((sample_names, labels), f)
            
        print(f"Saved {split_name}: {large_data.shape}")
        sys.stdout.flush()

    save_split(train_data_list, 'train')
    save_split(val_data_list, 'val')

if __name__ == "__main__":
    print("Inside Main Image Preprocess")
    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='data/bdsl_img')
    args = parser.parse_args()
    
    preprocess(args.data_root, args.output_dir)
