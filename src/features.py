import mediapipe as mp
import numpy as np
import cv2

class HandPreprocessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

    def extract_features(self, image):
        """
        Input: OpenCV Image (BGR)
        Output: List of lists of features [[wrist_y, pip_angle, pinky_abduction], ...]
        """
        # 1. Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return []
        
        features_list = []
        
        # Iterate through all detected hands
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (Left or Right)
            # Note: MediaPipe assumes mirrored input by default for selfies, but here we trust the output.
            # Label is 'Left' or 'Right'.
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
            else:
                handedness = 'Right' # Default fallback
            
            # Convert landmarks to numpy array (21 x 3)
            # [x, y, z]
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            # --- NORMALIZATION ---
            # 1. Center coordinates at the wrist (landmark 0)
            wrist = points[0].copy()
            points = points - wrist
            
            # 2. Normalize for handedness (Make Left hands look like Right hands)
            # If Left, flip the X-axis
            if handedness == 'Left':
                points[:, 0] = -points[:, 0]
                
            # --- EXTRACT FEATURES ---
            # Indices:
            # 0: Wrist
            # 9: Middle MCP
            # 10: Middle PIP
            # 12: Middle Tip
            # 17: Pinky MCP
            
            # Feature 1: Wrist Drop (Relative Y of Middle MCP vs Wrist)
            # Since wrist is at (0,0,0), this is just -points[9][1] (because y increases downwards)
            # Original logic: wrist.y - mcp.y.
            # Here: 0 - points[9][1] = -points[9][1].
            wrist_drop = -points[9][1]
            
            # Feature 2: Finger Curvature (Middle Finger PIP Angle)
            # Angle at PIP (10) formed by MCP(9)-PIP(10) and Tip(12)-PIP(10)
            pip_angle = self._calculate_angle(points[9], points[10], points[12])
            
            # Feature 3: Pinky Abduction
            # Angle between Middle Finger (Wrist->MCP) and Pinky (Wrist->MCP)
            # Since wrist is origin, vectors are points[9] and points[17]
            pinky_abduction = self._calculate_vector_angle(points[9], points[17])
            
            features_list.append([wrist_drop, pip_angle, pinky_abduction])

        return features_list

    def _calculate_angle(self, a, b, c):
        """Calculates angle at b (in degrees) given points a, b, c"""
        # Vector ba = a - b
        # Vector bc = c - b
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