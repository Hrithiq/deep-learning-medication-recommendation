# Deep Learning-Based Medication Recommendation System for Primary Care Decision Support

A multimodal deep learning framework for medication recommendation using clinical symptom information, patient demographics, comorbidities, drug-disease associations, and drug-drug interaction constraints.

> **Research / Academic Project:** This system is intended for academic research and decision-support experimentation. It is **not a clinical diagnostic or prescribing system** and must not be used to make real-world medical decisions.

---

## Overview

Medication recommendation in primary care requires consideration of multiple sources of information, including the patient's presenting symptoms, demographic characteristics, existing comorbidities, and potential interactions between medications.

The objective of this project is to develop a **Deep Learning-Based Medication Recommendation System for Primary Care Decision Support** that performs multi-label medication prediction while incorporating safety constraints.

The system combines:

* Clinical symptom information
* Patient demographics
* Comorbidities
* Disease-symptom associations
* Disease-drug associations
* Drug-drug interaction (DDI) knowledge

The primary prediction architecture uses a multimodal **FusionNet** that combines sequential symptom representations with structured patient features before performing multi-label medication prediction.

---

## System Architecture

```text
                  Clinical Inputs
                       │
          ┌────────────┼────────────┐
          │            │            │
      Symptoms        Age/Sex    Comorbidities
          │            │            │
          ▼            ▼            ▼
     Text Encoder   Structured Encoder
          │            │
          │            │
          └──────┬─────┘
                 │
                 ▼
             FusionNet
                 │
                 ▼
       Multi-Label Prediction
                 │
                 ▼
          Candidate Drugs
                 │
                 ▼
          DDI Safety Filter
                 │
                 ▼
        Top-K Recommendations
```

---

## Key Components

### 1. Synthetic Clinical Dataset Generation

A major challenge was the lack of a single dataset containing all required information in an aligned form.

The project therefore integrates information from multiple datasets and constructs a unified synthetic dataset.

The generation pipeline combines:

```text
Symptom → Disease knowledge
Disease → Drug knowledge
Drug → Drug Interaction knowledge
Patient demographic information
Comorbidity information
```

The resulting records contain fields such as:

```text
symptom_text
normalized_symptoms
age
sex
comorbidities
diagnosis
medications
```

The synthetic generation process uses stochastic sampling together with domain-based constraints rather than simply generating completely random records.

---

## Data Sources

The project uses multiple complementary sources.

### Symptom-Disease Dataset

Provides associations between diseases and their characteristic symptoms.

Example:

```text
Disease: Fungal infection

Symptoms:
- itching
- skin rash
- nodal skin eruptions
- dischromic patches
```

This information is used to construct:

```text
Disease → Symptoms
```

---

### Disease-Drug Dataset

Provides associations between diseases and medications.

This forms the treatment knowledge component:

```text
Disease → Drugs
```

---

### Drug-Drug Interaction Dataset

The DDI dataset contains interacting drug pairs and descriptions.

It is converted into a lookup structure:

```text
(drug_1, drug_2)
```

which is used during both dataset construction and medication safety filtering.

---

### Synthea

Synthea synthetic patient data is used as a source of realistic patient-level demographic and clinical information.

The project extracts only the attributes relevant to the proposed use case rather than using every available Synthea field.

---

## Data Preprocessing

The preprocessing pipeline performs several normalization operations.

### Text normalization

Clinical text is normalized using:

* Lowercasing
* Whitespace normalization
* Removal of irrelevant characters
* Standardization of terminology

---

### Drug normalization

Drug aliases are mapped to canonical names.

For example:

```text
Acetaminophen
        ↓
Paracetamol
```

This prevents the same medication from being represented as two different labels.

---

### Disease normalization

Disease terminology can differ across datasets.

For example:

```text
Acute bronchitis
Bronchitis
```

The preprocessing pipeline performs:

1. Exact matching
2. Substring matching
3. Fuzzy matching

to align disease terminology across datasets.

---

## Synthetic Data Generation

The synthetic dataset generator follows a rule-based stochastic generation process.

For each synthetic patient:

### Step 1 — Patient features

Patient characteristics are sampled:

```text
Age
Sex
Comorbidities
```

### Step 2 — Disease selection

One or more diseases are sampled from the available disease vocabulary.

### Step 3 — Symptom generation

Symptoms associated with the selected disease are sampled.

The system introduces controlled variation through symptom augmentation and different textual templates.

### Step 4 — Medication generation

Medications associated with the selected disease are retrieved from the disease-drug mapping.

### Step 5 — Safety filtering

Potential drug combinations are checked against the DDI knowledge base.

### Step 6 — Comorbidity rules

Medication recommendations can additionally be adjusted according to predefined comorbidity-related safety rules.

The final record combines all these components into a unified multimodal sample.

---

# Model

## FusionNet

FusionNet is the primary multimodal prediction architecture.

It contains two main branches:

```text
             Input
               │
       ┌───────┴────────┐
       │                │
 Symptom Sequence   Structured Data
       │                │
   Embedding          Linear Layer
       │                │
      LSTM              │
       │                │
       └───────┬────────┘
               │
          Concatenation
               │
          Fusion Layer
               │
        Drug Prediction
```

---

## Text Branch

Symptoms are converted into token IDs and passed through an embedding layer.

The embedding representation is then processed by an LSTM.

The LSTM produces a fixed-size representation of the symptom sequence.

This allows the model to learn relationships between symptoms rather than treating every symptom independently.

---

## Structured Branch

Structured patient information such as age, sex, and comorbidity representations is passed through a fully connected layer.

This converts heterogeneous structured features into a learned representation that can be combined with the symptom representation.

---

## Feature Fusion

The two representations are concatenated:

```text
Text representation
        +
Structured representation
        ↓
Joint clinical representation
```

The resulting representation is passed to the medication prediction layer.

---

# MPL — Multi-Label Prediction

Medication recommendation is naturally a **multi-label classification problem**.

A patient may require multiple medications simultaneously.

Therefore, the model produces one output score for each medication rather than selecting exactly one class.

The output layer produces:

```text
Drug 1 → probability
Drug 2 → probability
Drug 3 → probability
...
Drug N → probability
```

Sigmoid activation is used during interpretation because each medication can independently be considered as a candidate recommendation.

---

# Why Multi-Label Classification?

A traditional multi-class formulation assumes that only one class is correct.

That assumption does not hold for medication recommendation.

For example:

```text
Patient
   │
   ├── Drug A
   ├── Drug B
   └── Drug C
```

can all be relevant simultaneously.

Therefore, the task is formulated as multi-label prediction.

---

# DDI-Aware Learning

Medication recommendations must consider interactions between drugs.

A Drug-Drug Interaction matrix is constructed from the DDI dataset.

During training, an additional penalty discourages the model from assigning high probabilities to known interacting drug pairs.

Conceptually:

```text
Prediction Loss
      +
DDI Penalty
      ↓
Total Loss
```

This allows the model to optimize both predictive performance and medication safety constraints.

---

# Safety-Aware Top-K Decoding

After obtaining medication probabilities, the system selects the highest-ranked candidates.

However, rather than blindly selecting the top K drugs, candidate medications are checked against the DDI graph.

Conceptually:

```text
Predicted probabilities
        ↓
Sort by probability
        ↓
Candidate Drug 1
        ↓
DDI check
        ↓
Safe → Select
Unsafe → Reject
        ↓
Continue
        ↓
Top-K safe recommendations
```

This provides an additional safety layer during inference.

---

# Evaluation

The system evaluates medication recommendation using multi-label recommendation metrics.

### Jaccard Similarity

Measures overlap between predicted and ground-truth medication sets.

```text
Jaccard =
|Prediction ∩ Ground Truth|
---------------------------
|Prediction ∪ Ground Truth|
```

Higher values indicate greater overlap.

---

### DDI Rate

Measures the proportion of predicted medication pairs that correspond to known drug-drug interactions.

A lower DDI rate indicates safer recommendations.

---

### Additional Metrics

Depending on the experimental configuration, the project can also evaluate:

* Precision@K
* Recall@K
* F1@K
* MAP@K
* DDI rate

---

# Repository Structure

```text
deep-learning-medication-recommendation/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── Dataset_gen.ipynb
│   └── project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset_generation.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   └── inference.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── reports/
│   └── final_report.pdf
│
└── results/
    ├── figures/
    └── metrics/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/deep-learning-medication-recommendation.git
cd deep-learning-medication-recommendation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Generate the synthetic dataset

Open:

```text
notebooks/Dataset_gen.ipynb
```

Place the required source datasets in the appropriate data directory and execute the notebook.

The generated dataset should contain the required multimodal patient information and medication labels.

---

## 2. Train and evaluate the model

Open:

```text
notebooks/project.ipynb
```

Run the notebook sequentially to:

1. Load the processed dataset
2. Prepare model inputs
3. Train FusionNet
4. Apply the medication prediction layer
5. Apply DDI-aware constraints
6. Evaluate recommendations
7. Generate results

---

# Reproducibility

Experiments should use fixed random seeds where possible.

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

---

# Limitations

This project has several important limitations.

### Synthetic Data

The generated dataset is intended for controlled experimentation and does not replace real-world clinical data.

### Knowledge-base Coverage

The DDI database may not contain every possible drug interaction.

### Simplified Clinical Rules

Comorbidity-based medication adjustments are based on predefined rules and do not represent complete clinical guidelines.

### No Clinical Validation

The system has not been validated for use in real clinical environments.

Therefore, the model should be regarded as an academic decision-support prototype rather than a clinical prescribing tool.

---

# Future Work

Potential extensions include:

* Integration with larger clinical datasets
* Clinical-domain transformer encoders
* SciSpaCy-based medical entity recognition
* Graph neural networks for drug representation learning
* More comprehensive DDI knowledge graphs
* Temporal patient-history modelling
* Explainable AI using SHAP
* Calibration and uncertainty estimation
* External clinical validation
* Integration with standardized medical ontologies

---

# Ethical Considerations

Medication recommendation is a safety-critical application.

The system should therefore be evaluated for:

* Prediction errors
* Drug interaction coverage
* Dataset bias
* Demographic bias
* False recommendations
* Uncertainty
* Explainability

The system is intended only for academic experimentation and should not be used as a substitute for qualified medical professionals.

---

# Authors

**Hrithiq Gupta**
B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)
Manipal Institute of Technology, Manipal Academy of Higher Education

---

# Disclaimer

This repository contains an academic deep learning project for medication recommendation research.

**It is not a medical device, diagnostic system, or prescribing tool. Recommendations generated by the model must not be used for real-world medical treatment decisions.**
