import pandas as pd
from preprocessing import *
from synthetic_generator import SyntheticMedicalDataset

def build_dataset(n_samples):

    symptom_map = load_symptom_disease("data/symptom_disease.csv")
    drug_map = load_drug_disease("data/drug_disease.csv")
    ddi_map = load_ddi("data/ddi.csv")
    patients = load_patients("data/patients.csv")

    generator = SyntheticMedicalDataset(symptom_map, drug_map, ddi_map)

    data = []

    for i in range(n_samples):
        p = patients.sample(1).iloc[0]

        sample = generator.build_sample(
            age=p["AGE"],
            gender=p["GENDER"]
        )

        data.append(sample)

    df = pd.DataFrame(data)
    df.to_csv("final_dataset.csv", index=False)

    print("Dataset created:", df.shape)


if __name__ == "__main__":
    build_dataset(20000)