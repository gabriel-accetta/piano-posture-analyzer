import csv
import os
import cv2
from .features import BodyPreprocessor

def create_body_dataset():
	input_folder = "data/body_videos"
	output_file = "data/body_dataset.csv"
	processor = BodyPreprocessor()
	sample_every_n_frames = 5
	
	# Define columns
	header = [
		"label",
		"torso_inclination",
		"neck_angle",
		"shoulder_tension",
		"elbow_angle",
		"forearm_slope",
	]

	# Open CSV for writing
	with open(output_file, mode="w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header)

		# Loop through all videos in the folder
		for filename in os.listdir(input_folder):
			if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
				continue
			
			# --- AUTOMATIC LABELING LOGIC ---
            # Look for "class0", "class1", etc. in the filename
			try:
				# 'class0_correct.mp4' -> splits to ['class0', 'correct.mp4'] -> takes last char '0'
				label = int(filename.split("_")[0].replace("class", ""))
				print(f"Processing {filename} as Label {label}...")
			except Exception:
				print(f"Skipping {filename}: name must start with 'class0_', 'class1_', etc.")
				continue
			
			# Process Video
			cap = cv2.VideoCapture(os.path.join(input_folder, filename))
			frame_count = 0

			while cap.isOpened():
				success, frame = cap.read()
				if not success:
					break
				
				# Only process every n frames to avoid duplicate data
				if frame_count % sample_every_n_frames == 0:
					features = processor.extract_features(frame)
					if features:
						writer.writerow([label] + [float(x) for x in features])

				frame_count += 1

			cap.release()

	print(f"Done! Dataset saved to {output_file}")


if __name__ == "__main__":
	create_body_dataset()
