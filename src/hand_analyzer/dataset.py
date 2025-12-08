import cv2
import csv
import os
from .features import HandPreprocessor

def create_hand_dataset():
    input_folder = "data/hand_videos"
    output_file = "data/hand_dataset.csv"
    processor = HandPreprocessor()
    sample_every_n_frames = 5
    
    # Define columns
    header = [
        "label", 
        "wrist_y_rel", 
        "pip_angle_middle", 
        "dip_angle_middle", 
        "pip_angle_index", 
        "dip_angle_index", 
        "pip_angle_pinky", 
        "dip_angle_pinky"
    ]
    
    # Open CSV for writing
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Loop through all videos in the folder
        for filename in os.listdir(input_folder):
            if not filename.endswith((".mp4", ".mov", ".avi", ".mkv")): 
                continue
            
            # --- AUTOMATIC LABELING LOGIC ---
            # Look for "class0", "class1", etc. in the filename
            try:
                # 'class0_correct.mp4' -> splits to ['class0', 'correct.mp4'] -> takes last char '0'
                label = int(filename.split("_")[0].replace("class", ""))
                print(f"Processing {filename} as Label {label}...")
            except:
                print(f"Skipping {filename}: Name must start with 'class0_', 'class1_', etc.")
                continue

            # Process Video
            cap = cv2.VideoCapture(os.path.join(input_folder, filename))
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                # Only process every n frames to avoid duplicate data
                if frame_count % sample_every_n_frames == 0:
                    hands_pairs = processor.extract_features(frame)

                    # If hands were detected, write to CSV
                    if hands_pairs:
                        for handedness, features in hands_pairs:
                            writer.writerow([label] + features)
                        
                frame_count += 1
            
            cap.release()

    print(f"Done! Dataset saved to {output_file}")

if __name__ == "__main__":
    create_hand_dataset()