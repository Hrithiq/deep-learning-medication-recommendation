import numpy as np

# -------------------------------
# JACCARD
# -------------------------------
def jaccard(y_true, y_pred):
    intersection = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    return np.mean(intersection / (union + 1e-8))


# -------------------------------
# MAP@K
# -------------------------------
def apk(actual, predicted, k=5):
    score = 0.0
    hits = 0.0

    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(actual), k)


def mapk(y_true, y_pred, k=5):
    return np.mean([
        apk(a, p, k) for a, p in zip(y_true, y_pred)
    ])


# -------------------------------
# DDI RATE
# -------------------------------
def ddi_rate(preds, ddi_set, vocab):
    total = 0
    unsafe = 0

    for sample in preds:
        drugs = [vocab[i] for i in range(len(sample)) if sample[i] == 1]

        for i in range(len(drugs)):
            for j in range(i+1, len(drugs)):
                total += 1
                if (drugs[i], drugs[j]) in ddi_set:
                    unsafe += 1

    return unsafe / (total + 1e-8)