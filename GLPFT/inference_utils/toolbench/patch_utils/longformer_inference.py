import torch
from transformers import LongformerTokenizer

from utils.longformer_classifier import LongformerForSequenceClassification


class LongformerTextClassifier:
    def __init__(self, model_path='saved_models/categoriser', model_name='allenai/longformer-base-4096',
                 labels_file_path=''):
        # Load the tokenizer
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name, bf16=True)

        # Load the model
        self.model = LongformerForSequenceClassification(model_name, len(self.load_labels(labels_file_path)))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load the labels
        self.labels = self.load_labels(labels_file_path)

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_labels(self, labels_file_path):
        with open(labels_file_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def tokenize_text(self, text):
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )
        return encoded_input

    def predict(self, text):
        # Tokenize the input text
        encoded_input = self.tokenize_text(text)

        # Move input tensors to the same device as the model
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        with torch.no_grad():
            # Perform inference
            logits = self.model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(logits, dim=1)

        # Map the predicted index to the label
        predicted_label = self.labels[predicted.item()]

        return predicted_label