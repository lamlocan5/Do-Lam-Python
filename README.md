# APK Malware Detection Project

This project aims to detect malware in Android APK files using machine learning techniques, including Graph Convolutional Networks (GCN) and Random Forest classifiers.

## Project Structure

The project consists of several key components:

1. **Data Extraction** (`Extract.py`): Extracts permissions from APK files and processes them into a standardized format.
2. **Model Training** (`Train.py`): Trains a deep learning model using the CodeT5 architecture for permission-based classification.
3. **Inference** (`Inference.py`): Provides a Streamlit-based web interface for uploading and analyzing APK files.
4. **Graph-based Analysis** (`Inference.py`): Implements a GCN model for feature extraction from APK graphs.

## Setup and Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Extraction

Run the extraction script to process APK files:
