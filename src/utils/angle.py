import numpy as np

def calculate_angle(a, b, c):
        """Calculates angle at b (in degrees) given points a, b, c"""
        v1 = a - b
        v2 = c - b
        return calculate_vector_angle(v1, v2)

def calculate_vector_angle(v1, v2):
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