import mediapipe as mp
import numpy as np
import cv2
from ..utils.angle import calculate_angle, calculate_vector_angle

class BodyPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features(self, image):
        """
        Input: OpenCV Image (BGR)
        Output: List of lists of features [torso_inclination, neck_angle, shoulder_tension, elbow_angle, forearm_slope]
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_world_landmarks:
            return None

        # Extract World Landmarks (x, y, z in meters)
        lm = results.pose_world_landmarks.landmark

        # Convert landmarks to numpy array (33 x 3) of [x, y, z]
        points = np.array([[p.x, p.y, p.z] for p in lm])

        # --- VIRTUAL POINTS ---
        # Calculate mid-points to represent the "Spine" and "Head center"
        mid_shoulder = (points[11] + points[12]) / 2
        mid_hip = (points[23] + points[24]) / 2
        mid_ear = (points[7] + points[8]) / 2
        
        # --- NORMALIZATION ---
        # Scale factor: Torso Length (Mid-Hip to Mid-Shoulder)
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_length == 0: torso_length = 1.0

        # --- EXTRACT FEATURES ---
        # Indices:
        # 11,12: Shoulders (L,R)
        # 23,24: Hips (L,R)
        # 7,8: Ears (L,R)
        # 13,14: Elbows (L,R)
        # 15,16: Wrists (L,R)
        features = []

        # Feature 1: Torso Inclination
        # Vector from Hip to Shoulder
        spine_vector = mid_shoulder - mid_hip
        vertical_vector = np.array([0, -1, 0]) 
        torso_angle = calculate_vector_angle(spine_vector, vertical_vector)

        # Feature 2: Neck Angle
        neck_vector = mid_ear - mid_shoulder
        neck_angle = calculate_vector_angle(neck_vector, spine_vector)

        # Feature 3: Shoulder Tension
        # Average distance from ear to shoulder, normalized by torso
        l_trap_dist = np.linalg.norm(points[7] - points[11])
        r_trap_dist = np.linalg.norm(points[8] - points[12])
        avg_trap_dist = (l_trap_dist + r_trap_dist) / 2
        tension_ratio = avg_trap_dist / torso_length

        # 4. Elbow Angles (Arm Extension - Average of L and R)
        l_elbow_angle = calculate_angle(points[11], points[13], points[15])
        r_elbow_angle = calculate_angle(points[12], points[14], points[16])
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2

        # 5. Forearm Slope (Wrist Height vs Elbow Height)
        # Difference in Y between wrist and elbow, normalized
        # (Remember: in image coords, Y increases downwards, but World Y is metric)
        # Let's use relative height: (Wrist_Y - Elbow_Y)
        l_forearm_slope = (points[15][1] - points[13][1]) / torso_length
        r_forearm_slope = (points[16][1] - points[14][1]) / torso_length
        avg_forearm_slope = (l_forearm_slope + r_forearm_slope) / 2

        features = [torso_angle, neck_angle, tension_ratio, avg_elbow_angle, avg_forearm_slope]
        
        return features