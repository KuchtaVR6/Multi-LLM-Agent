import json
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, LongformerTokenizer
from tqdm import tqdm
import argparse
import sys
from utils.longformer_classifier import LongformerForSequenceClassification, load_labels, tokenize_text, save_labels


# Load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


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


def train(model, train_dataloader, optimizer, loss_fn, device, num_epochs, accumulation_steps):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            input_ids, attention_mask, targets = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs

            # Compute loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimization
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.empty_cache()

            if i % 5 == 0:  # Print loss every 100 batches
                print(f"Batch {i}/{len(train_dataloader)}, Loss: {loss.item()}")

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
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--labels_file_path", type=str, required=True, help="Path to the labels file.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    data = load_data(args.data_path)
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", bf16=True)
    inputs, targets = zip(*[(tokenize_text(item['input'], tokenizer), item['target']) for item in data])

    # Load labels
    labels = load_labels(args.labels_file_path)
    save_labels(labels, args.labels_file_path)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # Split data
    split_idx = int(0.9 * len(inputs))
    inputs_train, inputs_val = inputs[:split_idx], inputs[split_idx:]
    targets_train, targets_val = targets[:split_idx], targets[split_idx:]

    # Create datasets
    train_dataset = TextClassificationDataset(inputs_train, targets_train, label_to_id)
    val_dataset = TextClassificationDataset(inputs_val, targets_val, label_to_id)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the model
    model = LongformerForSequenceClassification("allenai/longformer-base-4096", len(labels))

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss()

    # Train the model
    train(model, train_dataloader, optimizer, loss_fn, device, args.num_epochs, args.accumulation_steps)

    # Evaluate
    accuracy = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the model
    save_model(model, args.model_save_path)


if __name__ == "__main__":
    # sys.argv = [
    #     'script_name.py',  # script name, ignored by argparse
    #     '--data_path', '/content/drive/MyDrive/Colab Notebooks/category_toy.json',
    #     '--batch_size', '2',
    #     '--num_epochs', '2',
    #     '--model_save_path', 'saved_models/categoriser',
    #     '--learning_rate', '5e-6',
    #     '--accumulation_steps', '4',
    #     '--labels_file_path', 'path_to_labels_file.txt'
    # ]

    main()
