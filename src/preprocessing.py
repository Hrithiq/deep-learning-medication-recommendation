import pandas as pd
import numpy as np
import re


DRUG_ALIAS = {
    "acetaminophen": "paracetamol",
    "tylenol": "paracetamol",
    "ibuprofen 200 mg oral tablet": "ibuprofen",
    "amoxicillin 250 mg / clavulanate 125 mg oral tablet": "amoxicillin_clavulanate"
}

def is_medical_condition(desc):
    desc = desc.lower()

    non_medical_keywords = [
        "housing",
        "education",
        "employment",
        "income",
        "social",
        "economic"
    ]

    for word in non_medical_keywords:
        if word in desc:
            return False

    return True

def load_conditions(path):
    df = pd.read_csv(path)

    df["DESCRIPTION"] = df["DESCRIPTION"].apply(clean_text)

    df = df[df["DESCRIPTION"].apply(is_medical_condition)]

    return df

def is_valid_medical(desc):
    return any(keyword in desc for keyword in [
        "disease", "disorder", "infection", "syndrome", "deficiency"
    ])

from difflib import get_close_matches

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

# -------------------------------
# CLEAN TEXT UTILS
# -------------------------------
def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s/]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_drug_name(x):
    x = clean_text(x)

    if x in DRUG_ALIAS:
        return DRUG_ALIAS[x]

    return x


# -------------------------------
# SYMPTOM → DISEASE MAP
# -------------------------------
def load_symptom_disease(path):
    df = pd.read_csv(path)

    disease_map = {}

    for _, row in df.iterrows():
        disease = clean_text(row["Disease"])

        symptoms = []
        for col in df.columns[1:]:
            val = row[col]
            if pd.notna(val):
                s = clean_text(val)
                if s:
                    symptoms.append(s)

        if disease not in disease_map:
            disease_map[disease] = set()

        disease_map[disease].update(symptoms)

    # convert to list
    disease_map = {k: list(v) for k, v in disease_map.items()}
    return disease_map


# -------------------------------
# DRUG → DISEASE MAP
# -------------------------------
def load_drug_disease(path):
    df = pd.read_csv(path)

    drug_map = {}

    for _, row in df.iterrows():
        disease = clean_text(row["disease"])
        drug = normalize_drug_name(row["drug"])

        if disease not in drug_map:
            drug_map[disease] = set()

        drug_map[disease].add(drug)

    return {k: list(v) for k, v in drug_map.items()}


# -------------------------------
# DDI MAP
# -------------------------------
def load_ddi(path):
    df = pd.read_csv(path)

    ddi_set = set()

    for _, row in df.iterrows():
        d1 = normalize_drug_name(row["Drug 1"])
        d2 = normalize_drug_name(row["Drug 2"])

        ddi_set.add((d1, d2))
        ddi_set.add((d2, d1))

    return ddi_set


# -------------------------------
# SYNTHETIC PATIENT FEATURES
# -------------------------------
def load_patients(path):
    df = pd.read_csv(path)

    df["AGE"] = 2026 - pd.to_datetime(df["BIRTHDATE"]).dt.year
    df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})

    return df[["Id", "AGE", "GENDER"]]