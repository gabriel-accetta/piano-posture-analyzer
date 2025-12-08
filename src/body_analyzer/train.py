import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    DATA_PATH = "data/body_dataset.csv"
    MODEL_PATH = "models/rf_body_posture_classifier.pkl"
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please run create_dataset.py first.")
        return

    # --- Data Preparation ---
    
    # Features (all columns except 'label')
    X = df.drop('label', axis=1)
    # Target (the 'label' column)
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")
    
    # --- Model Training ---
    
    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    print("Training Random Forest model...")
    rf_classifier.fit(X_train, y_train)
    
    # --- Model Evaluation ---
    
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # --- Model Saving ---
    
    joblib.dump(rf_classifier, MODEL_PATH)
    print(f"\nModel successfully trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
