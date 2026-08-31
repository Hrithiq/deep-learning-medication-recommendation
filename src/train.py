import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

class MedDataset(Dataset):
    def __init__(self, df, tokenizer, drug_vocab):
        self.df = df
        self.tokenizer = tokenizer
        self.drug_vocab = drug_vocab

    def __len__(self):
        return len(self.df)

    def encode_labels(self, meds):
        vec = [0]*len(self.drug_vocab)
        for m in meds:
            if m in self.drug_vocab:
                vec[self.drug_vocab.index(m)] = 1
        return torch.tensor(vec, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(
            row["symptom_text"],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        structured = torch.tensor([
            row["age"],
            1 if row["sex"] == "M" else 0,
            len(row["comorbidities"])
        ], dtype=torch.float)

        labels = self.encode_labels(eval(row["medications"]))

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "structured": structured,
            "labels": labels
        }


# TRAIN LOOP
def train():
    df = pd.read_csv("final_dataset.csv")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    drug_vocab = list(set(sum(df["medications"].apply(eval), [])))

    dataset = MedDataset(df, tokenizer, drug_vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MedRecModel(len(drug_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(5):
        for batch in loader:
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["structured"]
            )

            loss = loss_fn(out, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")


if __name__ == "__main__":
    train()