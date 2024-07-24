import torch
from transformers import LongformerModel
from torch import nn


class LongformerForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(LongformerForSequenceClassification, self).__init__()
        self.num_labels = num_labels

        # Load Longformer model
        self.longformer = LongformerModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        # Create a Linear layer
        self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)
        self.classifier = self.classifier.to(dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, token_type_ids=None):
        # Forward pass through Longformer
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids
        )
        # Take the pooled output (corresponding to [CLS] token)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


def load_labels(labels_file_path):
    with open(labels_file_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def save_labels(labels, labels_file_path):
    with open(labels_file_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")


def tokenize_text(text, tokenizer):
    encoded_input = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=4096,
        return_tensors='pt'
    )
    return encoded_input
