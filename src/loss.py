import torch

def ddi_penalty(preds, ddi_matrix):
    """
    preds: (batch, num_drugs)
    ddi_matrix: (num_drugs, num_drugs)
    """

    prob = torch.sigmoid(preds)

    pairwise = torch.matmul(prob.unsqueeze(2), prob.unsqueeze(1))
    ddi_loss = (pairwise * ddi_matrix).mean()

    return ddi_loss


def combined_loss(preds, targets, ddi_matrix, alpha=0.8):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)

    ddi = ddi_penalty(preds, ddi_matrix)

    return alpha * bce + (1 - alpha) * ddi