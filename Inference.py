import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

# Configuration
class Config:
    def __init__(self, model_name, num_labels, max_length, device, output_dir):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        self.output_dir = output_dir

# Define the model class
class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TextClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token
        logits = self.classifier(pooled_output)
        return logits

# Function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    model = TextClassifier(config.model_name, config.num_labels)
    model_path = os.path.join(config.output_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model, tokenizer

# Prediction function
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

# Streamlit app
def main():
    st.title("APK Permissions Classification")
    st.write("Enter the permissions text to classify whether it's benign or malware.")

    # User input
    user_input = st.text_area("Enter permissions text here", height=200)

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Classifying..."):
                # Load configuration
                config = Config(
                    model_name='bert-base-uncased',
                    num_labels=2,
                    max_length=256,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    output_dir='Checkpoints'
                )

                # Load model and tokenizer
                model, tokenizer = load_model_and_tokenizer(config)

                # Predict
                label, confidence = predict(model, tokenizer, config, user_input)

                # Map label to class name
                label_map = {0: 'Benign', 1: 'Malware'}
                result = label_map.get(label, "Unknown")

                st.success(f"Prediction: **{result}**")
                st.info(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
