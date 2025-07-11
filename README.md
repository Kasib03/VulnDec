# VulnDec

Detecting vulnerabilities in code snippets via a CNN + BiLSTM model with CodeBERT embeddings.
![Uploading image.png…]()


## 📌 Project Overview

This project presents a deep learning-based approach for **automatic code vulnerability detection and classification** using a hybrid model that integrates **CodeBERT embeddings** with a **CNN-BiLSTM** architecture. The model is designed to classify Python code snippets into **multiple vulnerability types** such as SQL Injection, XSS, XSRF, Remote Code Execution, and Open Redirect. It also supports a **dual-output** structure: one for predicting the *vulnerability type (language-level)* and another for predicting whether the code is *safe or vulnerable (binary classification)*.

## 🧠 Motivation

Traditional vulnerability detection tools rely heavily on static rules and manual reviews, making them inefficient for rapidly evolving codebases. Leveraging the semantic understanding of transformer-based models like **CodeBERT** with the sequential processing power of **CNNs and BiLSTMs**, this project aims to provide a more accurate and scalable solution for detecting vulnerabilities directly from source code.

## 🔍 Features

- ✅ Multi-class classification of 5 vulnerability types  
- ✅ Binary classification for safety status  
- ✅ CodeBERT for pre-trained code understanding  
- ✅ CNN layers for spatial feature extraction  
- ✅ BiLSTM layers for sequential dependencies  
- ✅ Stratified data splitting and class balancing  
- ✅ Evaluation metrics: F1-Score, Accuracy, Confusion Matrix  


## 🧬 Model Architecture

The architecture includes:

- **CodeBERT Layer**: Extracts token-level embeddings from code input  
- **CNN Block**: Applies convolution to capture local vulnerability patterns  
- **BiLSTM Block**: Captures long-term dependencies in token sequences  
- **Dense Output Layers**:
  - `language_output`: Softmax for 5-class vulnerability type
  - `safety_output`: Sigmoid for binary classification (safe/vulnerable)

## Repository Structure

- **Code/**
  - `cnn_bilstm_codebert_multiclass.py` – Main training script for the CNN + BiLSTM model
  - `myutils.py` – Utility functions (e.g., sliding-window preprocessing)
  - `vulnerable-code_20000.ipynb` - First 20000 words based on sliding window
  - `Vulnerable-Code_500.ipynb` - First 500 rows of the dataset 

- **Data/**  
  Datasets in JSON format, organized by vulnerability type.
  
  Link to CVEfixes dataset: https://zenodo.org/records/7029359

  Link to short dataset created(pre-processed): https://www.kaggle.com/datasets/erasez/cvefixes5 

- **Models/**  
  Saved model checkpoints and tokenizer files.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Kasib03/VulnDec.git
   cd VulnDec
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocess data (if needed)**  
   Raw JSON files are in `Data/`. Use utilities in `Code/myutils.py` for tokenization and sliding-window extraction.

2. **Run Each file seperately**


## File Descriptions

- **`cnn_bilstm_codebert_multiclass.py`**
  - Loads CodeBERT embeddings via Hugging Face.
  - Builds a Conv1D → MaxPool → BiLSTM → Dense architecture.
  - Handles both language-type and safety-type classification branches.
  - Plots training history with Matplotlib and Seaborn.

- **`myutils.py`**
  - Contains functions for loading JSON snippets.
  - Implements sliding-window token extraction for long code samples.


## 🙌 Acknowledgments

- [CodeXGLUE Dataset](https://github.com/microsoft/CodeXGLUE)
- [CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [VUDENC](https://github.com/LauraWartschinski/VulnerabilityDetection)
- Keras & HuggingFace Transformers



