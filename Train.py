import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class Config:
    def __init__(self, model_name, num_labels, learning_rate, batch_size, num_epochs, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class APKDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        permissions = item['normalized_sentence']
        
        # Tokenize permissions
        encoding = self.tokenizer(
            permissions,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert label (assuming binary classification)
        label = 1 if "malware" in item['apk_path'].lower() else 0

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class CodeSageClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.classifier = nn.Linear(2048, num_labels)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs[0][:, 0]  # Use the representation of the [CLS] token
        logits = self.classifier(pooled_output)
        return logits

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def train_model(config):
    device = config.device
    logging.info(f"Using device: {device}")

    # Load data
    logging.info("Loading data from APKPermissions.json")
    data = load_data('APKPermissions.json')
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    logging.info(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, add_eos_token=True)

    # Create datasets
    train_dataset = APKDataset(train_data, tokenizer, config.max_length)
    val_dataset = APKDataset(val_data, tokenizer, config.max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = CodeSageClassifier(config.model_name, config.num_labels).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}'):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{config.num_epochs}, Training loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        logging.info(f'Validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model

if __name__ == "__main__":
    config = Config(
        model_name='codesage/codesage-large',
        num_labels=2,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=10,
        max_length=512
    )

    try:
        model = train_model(config)
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise
