import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from disease_rules import DiseaseRules, create_disease_rules


# =====================
# SYMPTOMS AND DISEASES
# =====================
symptoms = [
    'fever', 'fatigue', 'headache', 'muscle_pain', 'chills', 'sweating',
    'dry_cough', 'shortness_of_breath', 'sore_throat', 'runny_nose', 'congestion',
    'loss_of_smell', 'loss_of_taste', 'sensitivity_to_light', 'sensitivity_to_sound',
    'stiff_neck', 'sinus_pressure',
    'nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'loss_of_appetite',
    'skin_rash', 'itching', 'pimples', 'blackheads', 'oily_skin', 'redness_of_eyes',
    'joint_pain', 'swelled_lymph_nodes', 'high_fever', 'mild_fever'
]

diseases = [
    'Migraine', 'Flu', 'COVID-19', 'Meningitis', 'Dengue', 'Common Cold', 'Acne'
]

def create_medical_dataset():
    disease_profiles = {
        'Migraine': {
            'required': ['headache'],
            'common': ['sensitivity_to_light', 'sensitivity_to_sound', 'nausea'],
            'sometimes': ['vomiting', 'sinus_pressure', 'fatigue']
        },
        'Flu': {
            'required': ['fever'],
            'common': ['muscle_pain', 'fatigue', 'headache', 'chills'],
            'sometimes': ['dry_cough', 'sore_throat', 'sweating']
        },
        'COVID-19': {
            'required': [],
            'common': ['fever', 'dry_cough', 'fatigue', 'loss_of_smell'],
            'sometimes': ['shortness_of_breath', 'loss_of_taste', 'headache']
        },
        'Meningitis': {
            'required': ['fever', 'headache'],
            'common': ['stiff_neck', 'nausea', 'sensitivity_to_light'],
            'sometimes': ['vomiting', 'fatigue', 'skin_rash']
        },
        'Dengue': {
            'required': ['high_fever'],
            'common': ['headache', 'joint_pain', 'skin_rash'],
            'sometimes': ['muscle_pain', 'swelled_lymph_nodes', 'nausea']
        },
        'Common Cold': {
            'required': [],
            'common': ['runny_nose', 'congestion', 'sore_throat'],
            'sometimes': ['mild_fever', 'fatigue', 'headache']
        },
        'Acne': {
            'required': ['pimples'],
            'common': ['blackheads', 'oily_skin'],
            'sometimes': ['skin_rash', 'itching', 'redness_of_eyes']
        }
    }

    samples = []
    for disease, profile in disease_profiles.items():
        for _ in range(150):  # 150 samples per disease
            sample = {symptom: 0 for symptom in symptoms}
            for symptom in profile['required']:
                sample[symptom] = 1
            for symptom in profile['common']:
                if np.random.random() < 0.8:
                    sample[symptom] = 1
            for symptom in profile['sometimes']:
                if np.random.random() < 0.3:
                    sample[symptom] = 1
            for symptom in symptoms:
                if (symptom not in profile['required'] + profile['common'] + profile['sometimes'] and
                    np.random.random() < 0.05):
                    sample[symptom] = 1
            sample['Disease'] = disease
            samples.append(sample)
    return pd.DataFrame(samples)

class DiseaseRules:
    @staticmethod
    def migraine(s):
        return 'headache' in s
    @staticmethod
    def flu(s):
        return 'fever' in s and ('muscle_pain' in s or 'fatigue' in s)
    @staticmethod
    def covid19(s):
        return 'fever' in s or 'dry_cough' in s or 'loss_of_smell' in s
    @staticmethod
    def meningitis(s):
        return 'fever' in s and 'headache' in s and 'stiff_neck' in s
    @staticmethod
    def dengue(s):
        return 'high_fever' in s and ('headache' in s or 'joint_pain' in s)
    @staticmethod
    def common_cold(s):
        return 'runny_nose' in s or 'congestion' in s or 'sore_throat' in s
    @staticmethod
    def acne(s):
        return 'pimples' in s

def create_disease_rules():
    return {
        'Migraine': DiseaseRules.migraine,
        'Flu': DiseaseRules.flu,
        'COVID-19': DiseaseRules.covid19,
        'Meningitis': DiseaseRules.meningitis,
        'Dengue': DiseaseRules.dengue,
        'Common Cold': DiseaseRules.common_cold,
        'Acne': DiseaseRules.acne
    }

# =====================
# MODEL TRAINING
# =====================
print("Generating dataset and training model...")
med_data = create_medical_dataset()
X = med_data.drop(columns=['Disease'])
y = med_data['Disease']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

print("Model trained. Saving to 'medical_model.pkl'...")

with open('medical_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'label_encoder': le,
        'feature_names': X.columns.tolist(),
        'symptoms_list': symptoms,
        'diseases_list': diseases,
        'disease_rules': create_disease_rules()
    }, f)

print("Done! File 'medical_model.pkl' is ready.")
