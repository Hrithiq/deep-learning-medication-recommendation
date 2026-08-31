import random
import re
import pandas as pd

from difflib import get_close_matches
def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s/]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_disease_name(disease, disease_vocab):
    disease = clean_text(disease)

    # exact match
    if disease in disease_vocab:
        return disease

    # substring match
    for d in disease_vocab:
        if d in disease or disease in d:
            return d

    # fuzzy match
    match = get_close_matches(disease, disease_vocab, n=1, cutoff=0.6)
    if match:
        return match[0]

    return None

class SyntheticMedicalDataset:

    def __init__(self, symptom_map, drug_map, ddi_map):
        self.symptom_map = symptom_map
        self.drug_map = drug_map
        self.ddi_map = ddi_map
        self.diseases = list(symptom_map.keys())

    def sample_diseases(self):
        return random.sample(self.diseases, k=random.randint(1, 2))

    def generate_symptoms(self, diseases):
        symptoms = []
        for d in diseases:
            if d in self.symptom_map:
                symptoms += random.sample(
                    self.symptom_map[d],
                    min(3, len(self.symptom_map[d]))
                )

        text = ", ".join(symptoms)

        # inject noise
        text = text.replace("fever", "fevr")
        text = text.replace("shortness of breath", "sob")

        return text, list(set(symptoms))

    def generate_medications(self, diseases):
        meds = []
        for d in diseases:

            norm_d = normalize_disease_name(d, self.drug_map.keys())

            if norm_d and norm_d in self.drug_map:
                meds += self.drug_map[norm_d]

        return list(set(meds))

    def apply_ddi_filter(self, meds):
        safe = []
        for m in meds:
            if all((m, s) not in self.ddi_map for s in safe):
                safe.append(m)
        return safe

    def build_sample(self, age, gender):
        diseases = self.sample_diseases()
        text, symptoms = self.generate_symptoms(diseases)
        meds = self.generate_medications(diseases)
        safe_meds = self.apply_ddi_filter(meds)

        return {
            "symptom_text": text,
            "normalized_symptoms": symptoms,
            "age": age,
            "sex": gender,
            "diagnosis": diseases,
            "medications": safe_meds
        }