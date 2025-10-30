import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
import string
import os

nltk.download('punkt', quiet=True)

# Load model and vectorizer once
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "skill_matching_model.pkl")
    vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Model load failed: {e}. Falling back to rule-based matching.")
    MODEL_AVAILABLE = False


def extract_skills(text):
    """Improved skill extraction that handles comma-separated and 'and'-separated inputs"""
    text = str(text).lower().strip()
    skills = []
    for part in text.split(','):
        skills.extend(part.split(' and '))
    return set(
        word.strip()
        for word in word_tokenize(' '.join(skills))
        if word.strip() and word not in string.punctuation
    )


def match_skills(users, project):
    """Skill matching using ML model (or rule-based fallback)"""
    project_text = project['requirements']

    if MODEL_AVAILABLE:
        try:
            # Vectorize both project requirement and user skills
            project_vec = vectorizer.transform([project_text])
            user_vecs = vectorizer.transform(users['skills'])

            # Predict match scores using ML model
            predicted_scores = model.predict(user_vecs)

            users = users.copy()
            users['match_score'] = predicted_scores

            # Sort by predicted score, then experience
            sorted_users = users.sort_values(
                by=['match_score', 'experience'], ascending=[False, False]
            ).reset_index(drop=True)

            return sorted_users[sorted_users['match_score'] > 0]
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}. Reverting to rule-based matching.")

    # === Rule-based fallback ===
    required_skills = extract_skills(project_text)
    if not required_skills:
        return pd.DataFrame()

    def calculate_score(user_skills):
        user_set = extract_skills(user_skills)
        if not required_skills:
            return 0
        return len(user_set & required_skills) / len(required_skills)

    users = users.copy()
    users['match_score'] = users['skills'].apply(calculate_score)

    sorted_users = users.sort_values(
        by=['match_score', 'experience'], ascending=[False, False]
    ).reset_index(drop=True)

    return sorted_users[sorted_users['match_score'] > 0]
