
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

# Define Keypoint Mapping
# Body (0-6)
# 0: Nose (MP 0)
# 1: R-Shoulder (MP 12)
# 2: L-Shoulder (MP 11)
# 3: R-Elbow (MP 14)
# 4: L-Elbow (MP 13)
# 5: R-Wrist (MP 16)
# 6: L-Wrist (MP 15)

BODY_INDICES = [0, 12, 11, 14, 13, 16, 15]

# Hand (Joints: Wrist, ThumbTip, IndexMCP, IndexTip, MidMCP, MidTip, RingMCP, RingTip, PinkyMCP, PinkyTip)
HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

def process_video(args):
    file_path, label, label_name = args
    
    mp_holistic = mp.solutions.holistic
    try:
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"Error init holistic: {e}")
        return None

    cap = cv2.VideoCapture(file_path)
    frames_data = []
    
    if not cap.isOpened():
        print(f"Error opening video: {file_path}")
        return None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
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
            
        frames_data.append(frame_kps)

    cap.release()
    holistic.close()
    
    if len(frames_data) == 0:
        return None

    # (T, C, V) -> (C, T, V)
    data = np.array(frames_data) 
    data = data.transpose(1, 0, 2) 
    
    # Add Person Dimension (M=1) -> (C, T, V, M)
    data = data[..., np.newaxis] 
    
    return (data, label, label_name, file_path)

def preprocess(dataset_root, output_dir, max_frames=300):
    print("Starting preprocess function")
    sys.stdout.flush()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Listing directory: {dataset_root}")
    sys.stdout.flush()
    classes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    
    # classes = classes[:1] 
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    sys.stdout.flush()
    
    all_tasks = []
    
    for cls_name in classes:
        cls_dir = os.path.join(dataset_root, cls_name)
        print(f"Scanning class {cls_name}...", end='\r')
        sys.stdout.flush()
        video_files = glob.glob(os.path.join(cls_dir, "*.mp4"))
        # print(f"Class {cls_name}: Found {len(video_files)} videos")
        # sys.stdout.flush()
        for video_file in video_files:
            all_tasks.append((video_file, class_to_idx[cls_name], cls_name))
            
    print(f"\nFound total {len(all_tasks)} videos.")
    sys.stdout.flush()
            
    print(f"Total videos to process: {len(all_tasks)}")
    sys.stdout.flush()

    results = []
    # for task in tqdm(all_tasks):
    for i, task in enumerate(all_tasks):
        print(f"Processing {i}/{len(all_tasks)}: {task[0]}")
        sys.stdout.flush()
        try:
            res = process_video(task)
            # if res is None:
            #     print(f"Failed to process (None result): {task[0]}")
            # else:
            #      print(f"Processed {task[0]}: {res[0].shape}")
            #      sys.stdout.flush()
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {task[0]}: {e}")
            sys.stdout.flush()
    
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} videos")
    sys.stdout.flush()
    
    # Split Train/Val (80/20)
    import random
    random.seed(42)
    random.shuffle(valid_results)
    
    split_idx = int(0.8 * len(valid_results))
    train_data_list = valid_results[:split_idx]
    val_data_list = valid_results[split_idx:]
    
    def save_split(data_list, split_name):
        if not data_list:
            return
            
        max_t = 0
        for item in data_list:
            max_t = max(max_t, item[0].shape[1])
        
        print(f"{split_name} Max T: {max_t}. Truncating/Padding to {max_frames} or keeping max?")
        final_T = max_frames
        
        N = len(data_list)
        C = 3
        V = 27
        M = 1
        
        large_data = np.zeros((N, C, final_T, V, M), dtype=np.float32)
        labels = []
        sample_names = []
        
        for i, (d, label, label_name, vid_path) in enumerate(data_list):
            t = d.shape[1]
            if t > final_T:
                large_data[i, :, :final_T, :, :] = d[:, :final_T, :, :]
            else:
                large_data[i, :, :t, :, :] = d
            
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
    print("Inside Main")
    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='data/bdsl')
    args = parser.parse_args()
    
    preprocess(args.data_root, args.output_dir)
