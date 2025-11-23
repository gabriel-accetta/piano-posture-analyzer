import cv2
import csv
import os
import sys
from src.features import HandPreprocessor

def create_hand_dataset():
    input_folder = "data/raw_videos"
    output_file = "data/dataset.csv"
    processor = HandPreprocessor()
    
    # Define columns: Label first, then your features
    # Make sure these match exactly what returns from extract_features in features.py
    header = ["label", "wrist_y_rel", "pip_angle_middle", "pinky_abduction"] 
    
    # 2. Open CSV for writing
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) # Write the header row
        
        # 3. Loop through all videos in the folder
        for filename in os.listdir(input_folder):
            if not filename.endswith(".mp4"): continue
            
            # --- AUTOMATIC LABELING LOGIC ---
            # We look for "class0", "class1", etc. in the filename
            try:
                # 'class0_correct.mp4' -> splits to ['class0', 'correct.mp4'] -> takes last char '0'
                label = int(filename.split("_")[0].replace("class", ""))
                print(f"Processing {filename} as Label {label}...")
            except:
                print(f"Skipping {filename}: Name must start with 'class0_', 'class1_', etc.")
                continue

            # 4. Process Video
            cap = cv2.VideoCapture(os.path.join(input_folder, filename))
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                # OPTIMIZATION: Only process every 5th frame to avoid duplicate data
                if frame_count % 5 == 0:
                    # Extract features using your shared logic
                    # Returns list of lists: [[f1, f2, f3], [f1, f2, f3]]
                    hands_features = processor.extract_features(frame)
                    
                    # If hands were detected, write to CSV
                    if hands_features:
                        for features in hands_features:
                            # Write [Label, Feature1, Feature2, Feature3...]
                            writer.writerow([label] + features)
                        
                frame_count += 1
            
            cap.release()

    print(f"Done! Dataset saved to {output_file}")

if __name__ == "__main__":
    create_hand_dataset()