import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from disease_rules import DiseaseRules, create_disease_rules


app = Flask(__name__)
app.secret_key = 'f2c559b34e5c13912fa23068bb7a00e0ab998dee218b9cbdf7cfdefde6aa6d4a'  # Replace with a secure key

# Load model and metadata
with open('medical_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

SYMPTOMS = list(model_data['feature_names'])
DISEASES = model_data['diseases_list']

def predict_disease(symptoms_input):
    present_symptoms = set()
    input_vector = np.zeros(len(SYMPTOMS))
    for symptom in symptoms_input:
        symptom = symptom.strip().lower().replace(' ', '_')
        if symptom in SYMPTOMS:
            idx = SYMPTOMS.index(symptom)
            input_vector[idx] = 1
            present_symptoms.add(symptom)
    if not present_symptoms:
        return None, "No valid symptoms provided"
    disease_probs = model_data['model'].predict_proba([input_vector])[0]
    disease_rules = model_data['disease_rules']
    valid_predictions = []
    for i, prob in enumerate(disease_probs):
        disease = model_data['label_encoder'].classes_[i]
        if disease_rules[disease](present_symptoms):
            valid_predictions.append((disease, prob))
    valid_predictions.sort(key=lambda x: -x[1])
    return valid_predictions[:3], None

def get_recommendations(disease):
    precautions_db = {
        'Migraine': [
            "Rest in a quiet, dark room",
            "Apply cold compress to forehead or neck",
            "Stay hydrated and avoid triggers",
            "Consider over-the-counter pain relievers",
            "Practice stress management techniques"
        ],
        'Flu': [
            "Get plenty of rest",
            "Stay hydrated with water and electrolyte drinks",
            "Use fever reducers like acetaminophen",
            "Use a humidifier to ease breathing",
            "Stay home to avoid spreading the virus"
        ],
        'COVID-19': [
            "Isolate from others",
            "Monitor oxygen levels if available",
            "Rest and stay hydrated",
            "Use over-the-counter medications for symptom relief",
            "Seek medical attention if breathing difficulties develop"
        ],
        'Meningitis': [
            "SEEK EMERGENCY MEDICAL CARE IMMEDIATELY",
            "Do not wait for symptoms to worsen",
            "Follow all instructions from healthcare providers",
            "Rest in a quiet environment while awaiting care",
            "Avoid bright lights which may worsen symptoms"
        ],
        'Dengue': [
            "Get plenty of rest",
            "Stay hydrated with oral rehydration solutions",
            "Use acetaminophen for pain/fever (avoid aspirin/NSAIDs)",
            "Monitor for warning signs like severe abdominal pain",
            "Seek medical care immediately if symptoms worsen"
        ],
        'Common Cold': [
            "Rest and stay hydrated",
            "Use saline nasal sprays or rinses",
            "Gargle with warm salt water for sore throat",
            "Use a humidifier to ease congestion",
            "Get plenty of sleep to support immune function"
        ],
        'Acne': [
            "Wash face gently twice daily with mild cleanser",
            "Avoid picking or squeezing pimples",
            "Use oil-free, non-comedogenic skincare products",
            "Apply topical treatments as directed",
            "See dermatologist if acne is severe or persistent"
        ]
    }
    medications_db = {
        'Migraine': [
            "Ibuprofen (Advil, Motrin)",
            "Acetaminophen (Tylenol)",
            "Sumatriptan (Imitrex) for severe cases",
            "Anti-nausea medications if needed"
        ],
        'Flu': [
            "Oseltamivir (Tamiflu) - antiviral",
            "Acetaminophen (Tylenol) - fever/pain",
            "Dextromethorphan - cough suppressant",
            "Saline nasal spray for congestion"
        ],
        'COVID-19': [
            "Paxlovid (nirmatrelvir/ritonavir) - antiviral",
            "Acetaminophen (Tylenol) - fever/pain",
            "Dextromethorphan - cough suppressant",
            "Guaifenesin - expectorant"
        ],
        'Meningitis': [
            "Intravenous antibiotics (ceftriaxone, vancomycin)",
            "Corticosteroids to reduce inflammation",
            "Anticonvulsants if seizures occur",
            "Pain and fever medications"
        ],
        'Dengue': [
            "Acetaminophen (Tylenol) for pain/fever",
            "Oral rehydration solutions",
            "Intravenous fluids in severe cases",
            "Avoid aspirin/NSAIDs due to bleeding risk"
        ],
        'Common Cold': [
            "Acetaminophen (Tylenol) - pain/fever",
            "Pseudoephedrine - decongestant",
            "Dextromethorphan - cough suppressant",
            "Guaifenesin - expectorant"
        ],
        'Acne': [
            "Benzoyl peroxide - antibacterial",
            "Salicylic acid - exfoliant",
            "Adapalene (Differin) - retinoid",
            "Clindamycin - antibiotic"
        ]
    }
    advice = {
        'Migraine': "Seek immediate care if: sudden severe headache, headache after injury, or with fever/stiff neck/confusion.",
        'Flu': "See a doctor if: difficulty breathing, persistent fever, or symptoms worsen after 3-4 days.",
        'COVID-19': "Seek emergency care for: trouble breathing, persistent chest pain, confusion, or pale/bluish lips/face.",
        'Meningitis': "THIS IS A MEDICAL EMERGENCY. Seek immediate care for fever with headache and stiff neck.",
        'Dengue': "Seek immediate care if: severe abdominal pain, persistent vomiting, bleeding, or restlessness.",
        'Common Cold': "See a doctor if: symptoms last >10 days, fever >101°F, or difficulty breathing.",
        'Acne': "See a dermatologist if: painful cysts, scarring, or no improvement after 2-3 months of treatment."
    }
    return {
        'precautions': precautions_db.get(disease, []),
        'medications': medications_db.get(disease, []),
        'when_to_see_doctor': advice.get(disease, "Consult a healthcare professional if symptoms persist or worsen.")
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected = request.form.getlist('symptoms')
        if not selected:
            flash('Please select at least one symptom.', 'danger')
            return redirect(url_for('index'))
        predictions, error = predict_disease(selected)
        if error:
            flash(error, 'danger')
            return redirect(url_for('index'))
        if not predictions:
            flash('No valid diagnoses match your symptoms. Please check your entries or consult a doctor.', 'warning')
            return redirect(url_for('index'))
        results = []
        for disease, prob in predictions:
            rec = get_recommendations(disease)
            results.append({
                'disease': disease,
                'prob': f"{prob*100:.1f}",
                'precautions': rec['precautions'][:3],
                'medications': rec['medications'][:3],
                'doctor_advice': rec['when_to_see_doctor']
            })
        return render_template('result.html', results=results, selected=selected)
    return render_template('index.html', symptoms=SYMPTOMS)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
