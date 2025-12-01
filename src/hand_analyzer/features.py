import mediapipe as mp
import numpy as np
import cv2

class HandPreprocessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            model_complexity=1
        )

    def extract_features(self, image):
        """
        Input: OpenCV Image (BGR)
        Output: List of lists of features [[wrist_y, pip_angle, dip_angle], ...]
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return []
        
        features_list = []
        
        # Iterate through all detected hands
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness ("Left" or "Right")
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
            else:
                handedness = 'Right'
            
            # Convert landmarks to numpy array (21 x 3) of [x, y, z]
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            # --- NORMALIZATION ---
            # 1. Center coordinates at the wrist (landmark 0)
            wrist = points[0].copy()
            points = points - wrist
            
            # 2. Calculate Scale Factor (Distance from Wrist to Middle MCP)
            # Since points are centered at wrist, Wrist is (0,0,0)
            # So distance is just norm of Middle MCP (index 9)
            scale_factor = np.linalg.norm(points[9])
            if scale_factor == 0: scale_factor = 1.0 # Avoid division by zero
            
            # 3. Normalize for handedness (Make Left hands look like Right hands)
            # If Left, flip the X-axis
            if handedness == 'Left':
                points[:, 0] = -points[:, 0]
                
            # --- EXTRACT FEATURES ---
            # Indices:
            # 0: Wrist
            # 5, 6, 7, 8: Index (MCP, PIP, DIP, Tip)
            # 9, 10, 11, 12: Middle (MCP, PIP, DIP, Tip)
            # 17, 18, 19, 20: Pinky (MCP, PIP, DIP, Tip)
            
            # Feature 1: Wrist Drop (Relative Y of Middle MCP vs Wrist)
            # Normalized by scale factor
            wrist_drop = (-points[9][1]) / scale_factor
            
            # Feature 2: Middle Finger Curvature
            middle_pip = self._calculate_angle(points[9], points[10], points[12])
            middle_dip = self._calculate_angle(points[10], points[11], points[12])
            
            # Feature 3: Index Finger Curvature
            index_pip = self._calculate_angle(points[5], points[6], points[8])
            index_dip = self._calculate_angle(points[6], points[7], points[8])

            # Feature 4: Pinky Finger Curvature
            pinky_pip = self._calculate_angle(points[17], points[18], points[20])
            pinky_dip = self._calculate_angle(points[18], points[19], points[20])
            
            features_list.append([wrist_drop, middle_pip, middle_dip, index_pip, index_dip, pinky_pip, pinky_dip])

        return features_list

    def _calculate_angle(self, a, b, c):
        """Calculates angle at b (in degrees) given points a, b, c"""
        v1 = a - b
        v2 = c - b
        return self._calculate_vector_angle(v1, v2)

    def _calculate_vector_angle(self, v1, v2):
        """Calculates angle between two vectors in degrees"""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        
        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))