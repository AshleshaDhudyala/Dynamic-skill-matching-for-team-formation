import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize

# ✅ Safe conditional download of the punkt tokenizer
def ensure_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def extract_skills(text):
    """Improved skill extraction that handles comma-separated and 'and'-separated skills."""
    ensure_punkt()  # Make sure punkt is available before tokenizing

    if pd.isna(text):
        return set()

    text = text.lower().strip()

    # Split by comma and 'and'
    skills = []
    for part in text.split(','):
        skills.extend(part.split(' and '))

    # Tokenize and clean
    return set(
        word.strip()
        for word in word_tokenize(' '.join(skills))
        if word.strip() and word not in string.punctuation
    )

def match_skills(users, project):
    """Precise skill matching based on project requirements."""
    required_skills = extract_skills(project['requirements'])

    if not required_skills:
        return pd.DataFrame()  # Return empty if no project requirements

    def calculate_score(user_skills):
        user_skills_set = extract_skills(user_skills)
        return len(user_skills_set & required_skills) / len(required_skills) if required_skills else 0

    users = users.copy()
    users['match_score'] = users['skills'].apply(calculate_score)

    # Sort by match score and experience
    sorted_users = users.sort_values(
        by=['match_score', 'experience'],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Return only those with non-zero match score
    return sorted_users[sorted_users['match_score'] > 0]

