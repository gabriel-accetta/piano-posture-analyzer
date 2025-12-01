import cv2
import joblib
import numpy as np
from .features import HandPreprocessor

# Load model outside the function to avoid reloading it on every frame
try:
    MODEL = joblib.load("models/rf_hand_posture_classifier.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure train.py has been run.")
    MODEL = None
    
POSTURE_LABELS = {
    0: "Correct Posture",
    1: "Flat Fingers",
    2: "High Wrist",
    3: "Dropped Wrist",
    4: "Collapsed Joints"
}

def classify_posture(features_array):
    """
    Infers the posture class based on the 7 input features.
    
    Returns: A tuple (int_label, string_description)
    """
    if MODEL is None:
        return -1, "MODEL_NOT_LOADED"
        
    # Predict the class
    prediction = MODEL.predict(features_array)[0]
    
    # Return the result
    return prediction, POSTURE_LABELS.get(prediction, "UNKNOWN")

def run_live_inference():
    """Handles camera capture and real-time feature extraction/classification."""
    
    if MODEL is None:
        return
        
    processor = HandPreprocessor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting real-time posture analyzer. Press 'q' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        
        # Extract Features from the frame
        hands_pairs = processor.extract_features(image)

        # Classify each detected hand
        if hands_pairs:
            for i, (handedness, features) in enumerate(hands_pairs):

                # Unpack the 7 features directly for the classify_posture function call
                wrist_drop, middle_pip, middle_dip, index_pip, index_dip, pinky_pip, pinky_dip = features

                # Call the classifier function
                predicted_label, label_text = classify_posture(
                    np.array([features])
                )

                # Display Results
                cv2.putText(
                    image,
                    f"Hand {i+1} ({handedness}): {label_text}",
                    (20, 30 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow('Piano Hand Posture Analyzer MVP', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()