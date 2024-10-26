import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Set environment variables before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,garbage_collection_threshold:0.6,expandable_segments:True'

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
    def __init__(self, model_name, num_labels, learning_rate, batch_size, num_epochs, max_length, data_path, device, output_dir):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.data_path = data_path
        self.device = device
        self.output_dir = output_dir  # Directory to save the model and tokenizer

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

class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # Use the representation of the [CLS] token
        logits = self.classifier(pooled_output)
        return logits

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def train_model(config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load and prepare data
    data = load_data(config.data_path)
    
    # Split data into train and validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Load model
    model = TextClassifier(config.model_name, config.num_labels).to(config.device)
    
    # Enable gradient checkpointing
    model.encoder.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    # Create datasets
    train_dataset = APKDataset(train_data, tokenizer, config.max_length)
    val_dataset = APKDataset(val_data, tokenizer, config.max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Set up gradient accumulation
    gradient_accumulation_steps = 16  # Adjust as needed

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        total_loss = 0
        for step, batch in enumerate(train_loop):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)

            # Mixed precision training
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps

            # Gradient accumulation
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            train_loop.set_postfix(loss=loss.item() * gradient_accumulation_steps)

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{config.num_epochs}, Training loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        logging.info(f'Validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Save the trained model
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    model_save_path = os.path.join(config.output_dir, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info(f'Model saved to {model_save_path}')

    # Save the tokenizer
    tokenizer.save_pretrained(config.output_dir)
    logging.info(f'Tokenizer saved to {config.output_dir}')

    return model

def load_model(config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)

    # Load model architecture
    model = TextClassifier(config.model_name, config.num_labels)
    model_load_path = os.path.join(config.output_dir, 'model.pth')
    model.load_state_dict(torch.load(model_load_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    logging.info(f'Model loaded from {model_load_path}')

    return model, tokenizer

def predict(model, tokenizer, config, text):
    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    label = predicted.item()
    confidence = confidence.item()
    return label, confidence

if __name__ == "__main__":
    config = Config(
        model_name='bert-base-uncased',  # Switched to BERT-base
        num_labels=2,
        learning_rate=2e-5,
        batch_size=1,  # Keep batch size at 1
        num_epochs=10,
        max_length=256,  # Adjust as needed
        data_path='APKPermissions.json',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='Checkpoints'  # Directory to save the model and tokenizer
    )

    # Clear cache before training
    torch.cuda.empty_cache()

    try:
        model = train_model(config)
        logging.info("Training completed successfully!")

        # Load the model for inference
        model, tokenizer = load_model(config)

        # Example inference
        sample_text = "android.permission.READ_CONTACTS"
        label, confidence = predict(model, tokenizer, config, sample_text)
        logging.info(f'Predicted label: {label}, Confidence: {confidence:.4f}')
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise
