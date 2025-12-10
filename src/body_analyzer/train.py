import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    
    # --- Generate Visualizations ---
    plot_training_charts(rf_classifier, X_test, y_test, y_pred, accuracy)
    
    # --- Model Saving ---
    
    joblib.dump(rf_classifier, MODEL_PATH)
    print(f"\nModel successfully trained and saved to {MODEL_PATH}")


def plot_training_charts(model, X_test, y_test, y_pred, accuracy):
    """Generate and save training visualization charts."""
    
    # Ensure output directory exists
    os.makedirs('models/charts', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Body Posture Classifier')
    plt.tight_layout()
    plt.savefig('models/charts/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to models/charts/confusion_matrix.png")
    plt.close()
    
    # 3. Feature Importance (Top 15)
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    
    indices = np.argsort(feature_importance)[-15:]
    top_features = feature_names[indices]
    top_importances = feature_importance[indices]
    
    ax.barh(range(len(top_features)), top_importances, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Most Important Features - Body Posture')
    plt.tight_layout()
    plt.savefig('models/charts/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance chart saved to models/charts/feature_importance.png")
    plt.close()
    
    # 4. Model Performance Summary
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Calculate per-class accuracy
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY - BODY POSTURE CLASSIFIER
    
    Overall Accuracy: {accuracy:.4f}
    Test Set Size: {len(y_test)} samples
    
    Per-Class Accuracy:
    """
    
    for c in np.unique(y_test):
        mask = y_test == c
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        summary_text += f"\n  Class {c}: {class_acc:.4f}"
    
    summary_text += f"\n\nModel Parameters:\n  Estimators: 100\n  Max Depth: 10\n  Random State: 42"
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('models/charts/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Performance summary saved to models/charts/performance_summary.png")
    plt.close()
    
    print("\n✓ All training charts generated successfully!")


if __name__ == "__main__":
    train_model()
