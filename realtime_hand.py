import cv2
import sys
import os
import mediapipe as mp
import numpy as np

# Add src to path so we can import features
sys.path.append(os.path.abspath('src'))

from features import HandPreprocessor

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize preprocessor
    preprocessor = HandPreprocessor()
    
    # MediaPipe drawing utils
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        # 1. Extract features using our custom class
        # Note: extract_features expects BGR image, which is what opencv provides
        features_pairs = preprocessor.extract_features(frame)
        
        # 2. Get landmarks for visualization
        # We process the image again to get landmarks for drawing. 
        # Ideally, we would modify HandPreprocessor to return landmarks too, 
        # but we stick to the existing interface for now.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = preprocessor.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Annotate with features if available for this hand (list of pairs)
                if idx < len(features_pairs):
                    handedness, feat = features_pairs[idx]
                    wrist_drop, m_pip, m_dip, i_pip, i_dip, p_pip, p_dip = feat
                    
                    # Calculate text position (near the wrist)
                    h, w, _ = frame.shape
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x = int(wrist.x * w)
                    wrist_y = int(wrist.y * h)
                    
                    # Get handedness
                    # If the preprocessor returned handedness, prefer it; otherwise fall back
                    # to MediaPipe's handedness info from results (if present).
                    if not handedness:
                        if results.multi_handedness:
                            handedness = results.multi_handedness[idx].classification[0].label
                        else:
                            handedness = 'Right'
                    
                    # Prepare text
                    info_text = [
                        f"Wrist Drop: {wrist_drop:.3f}",
                        f"M. PIP: {m_pip:.0f} | DIP: {m_dip:.0f}",
                        f"I. PIP: {i_pip:.0f} | DIP: {i_dip:.0f}",
                        f"P. PIP: {p_pip:.0f} | DIP: {p_dip:.0f}",
                        f"Hand: {handedness}"
                    ]
                    
                    # Draw text background and text
                    for i, line in enumerate(info_text):
                        text_pos = (max(10, wrist_x - 100), max(20, wrist_y + 30 + (i * 25)))
                        
                        # Draw black outline for better visibility
                        cv2.putText(
                            frame, 
                            line, 
                            text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 0, 0), # Black
                            4, 
                            cv2.LINE_AA
                        )
                        # Draw text
                        cv2.putText(
                            frame, 
                            line, 
                            text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (255, 255, 0), # Cyan/Yellowish
                            2, 
                            cv2.LINE_AA
                        )

        # Display the resulting frame
        cv2.imshow('Realtime Hand Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
