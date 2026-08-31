import re

ABBREVIATIONS = {
    "sob": "shortness of breath",
    "fevr": "fever",
}

def normalize_text(text):
    text = text.lower()

    # expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)

    return text


def simple_ner(text, symptom_vocab):
    extracted = []
    for symptom in symptom_vocab:
        if symptom in text:
            extracted.append(symptom)
    return extracted