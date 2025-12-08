import cv2
import joblib
import numpy as np
import pandas as pd
from .features import BodyPreprocessor

# Load model once to avoid repeated disk I/O
try:
    MODEL = joblib.load("models/rf_body_posture_classifier.pkl")
    print("Body model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure train.py has been run.")
    MODEL = None
    
POSTURE_LABELS = {
    0: "Correct",
    1: "Slouched",
    2: "Head Forward",
    3: "Shoulders Raised",
    4: "Elbow Dropped",
    5: "Elbow Raised"
}

FEATURE_NAMES = [
    'torso_inclination',
    'neck_angle',
    'shoulder_tension',
    'elbow_angle',
    'forearm_slope'
]


def classify_posture(features_array):
    """
    Infers the body posture class based on the input features (5 features).

    Returns: (int_label, string_description)
    """
    if MODEL is None:
        return -1, "MODEL_NOT_LOADED"

    # Accept either a single-row array-like or many rows
    features_df = pd.DataFrame(features_array, columns=FEATURE_NAMES)

    prediction = MODEL.predict(features_df)[0]

    return prediction, POSTURE_LABELS.get(prediction, "UNKNOWN")


def run_live_inference():
    """Runs webcam loop, extracts body features and overlays classification."""
    if MODEL is None:
        print("Model not loaded — aborting live inference.")
        return

    processor = BodyPreprocessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time body posture analyzer. Press 'q' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)

        features = processor.extract_features(image)

        if features is not None:
            predicted_label, label_text = classify_posture(np.array([features]))

            cv2.putText(
                image,
                f"Posture: {label_text}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow('Piano Body Posture Analyzer', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_inference()
