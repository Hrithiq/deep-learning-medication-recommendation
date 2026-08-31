import torch
import torch.nn as nn
from transformers import AutoModel

class MedRecModel(nn.Module):

    def __init__(self, num_drugs):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        self.fc_struct = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_drugs),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, structured):
        text_emb = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state[:,0]

        struct_emb = self.fc_struct(structured)

        combined = torch.cat([text_emb, struct_emb], dim=1)

        return self.classifier(combined)