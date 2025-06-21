import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import os

# NLTK data initialization
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)  # Ensure directory exists
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.data.path.append(nltk_data_path)


def extract_skills(text):
    """Improved skill extraction that handles comma-separated skills and filters out non-skill tokens"""
    text = text.lower().strip()
    # Handle both comma and "and" separated skills
    skills = []
    for part in text.split(','):
        skills.extend(part.split(' and '))
    # Clean and tokenize, removing non-skill tokens
    return set(word.strip() for word in word_tokenize(' '.join(skills)) if word.strip() and word not in string.punctuation)

def match_skills(users, project):
    """Precise skill matching that respects project requirements"""
    # Get required skills from project
    required_skills = extract_skills(project['requirements'])
    
    if not required_skills:
        return pd.DataFrame()  # If no required skills are specified, return an empty DataFrame
    
    # Calculate match score for each user
    def calculate_score(user_skills):
        user_skills_set = extract_skills(user_skills)
        # Percentage of required skills matched
        if not required_skills:
            return 0
        return len(user_skills_set & required_skills) / len(required_skills)
    
    users['match_score'] = users['skills'].apply(calculate_score)
    
    # Sort by match score (descending) then experience (descending)
    sorted_users = users.sort_values(
        by=['match_score', 'experience'], 
        ascending=[False, False]
    ).reset_index(drop=True)

    # Filter out users with zero match score
    sorted_users = sorted_users[sorted_users['match_score'] > 0]
    
    return sorted_users




