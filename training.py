import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_model():
    """Train, evaluate, and save the skill matching model"""

    # Get file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "training_data.csv")
    model_path = os.path.join(current_dir, "skill_matching_model.pkl")
    vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

    # Load training data
    training_data = pd.read_csv(training_path)

    # Prepare features and labels
    X_text = training_data['skills']
    y = training_data['match_score']

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_text)

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # === Visualization ===
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.bar(["Accuracy"], [accuracy * 100], color="green")
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Model Accuracy")
    plt.tight_layout()
    plt.show()

    print("🎯 Training complete and visualizations displayed.")

if __name__ == "__main__":
    train_model()
