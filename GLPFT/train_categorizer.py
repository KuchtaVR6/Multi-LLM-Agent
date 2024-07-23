import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerTokenizer, AdamW
from tqdm import tqdm
import argparse
from utils.data_split import train_validation_split
import numpy as np


# Load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# Tokenize the data
def tokenize_data(data, tokenizer):
    inputs = []
    targets = []
    for item in tqdm(data):
        encoded_input = tokenizer(
            item['input'],
            padding='max_length',
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )
        inputs.append(encoded_input)
        targets.append(item['target'])
    return np.array(inputs), np.array(targets)


class TextClassificationDataset(Dataset):
    def __init__(self, inputs, targets, label_to_id):
        self.inputs = inputs
        self.targets = targets
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]['input_ids'].squeeze().long()
        attention_mask = self.inputs[idx]['attention_mask'].squeeze().to(torch.bool)
        target = torch.tensor(self.label_to_id[self.targets[idx]], dtype=torch.long)
        return input_ids, attention_mask, target


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, targets = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs

            # Predictions
            _, predicted = torch.max(logits, dim=1)

            # Calculate accuracy
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    return accuracy


def train(model, train_dataloader, optimizer, loss_fn, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, targets = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs

            # Compute loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Longformer model for text classification.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--model_save_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    data = load_data(args.data_path)
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", bf16=True)
    inputs, targets = tokenize_data(data, tokenizer)

    # Map labels to IDs
    unique_labels = list(set(targets))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    # Split data
    inputs_train, inputs_val, targets_train, targets_val = train_validation_split(inputs, targets, test_size=0.1)

    # Create datasets
    train_dataset = TextClassificationDataset(inputs_train, targets_train, label_to_id)
    val_dataset = TextClassificationDataset(inputs_val, targets_val, label_to_id)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the model
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
            # Pass it through the classification head
            logits = self.classifier(pooled_output)
            return logits

    num_labels = len(label_to_id)
    model_name = "allenai/longformer-base-4096"
    model = LongformerForSequenceClassification(model_name, num_labels)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss()

    # Train the model
    train(model, train_dataloader, optimizer, loss_fn, device, args.num_epochs)

    # Evaluate
    accuracy = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the model
    save_model(model, args.model_save_path)


if __name__ == "__main__":
    main()