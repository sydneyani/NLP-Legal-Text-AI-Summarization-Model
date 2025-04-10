# main.py
# AI-Powered Legal Document Summarization
# Team Members: Sydney Ani (T5 Abstractive) and Nicolas Osorio (BERTSUM Extractive)

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

# Ensure paths are correct
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Simple imports for your folder structure
from dataset.data_utils import load_data, preprocess_data, analyze_dataset
from abstractive.t5_summarizer import T5LegalSummarizer
from extractive.bertsum_extractor import BERTSumExtractor

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Download NLTK resources
nltk.download('punkt')

def main():
    # Step 1: Load and analyze the dataset
    print("Step 1: Loading and analyzing the dataset...")
    
    # Try to load the actual dataset
    try:
        df = load_data()
        if df is None:
            # Fallback to sample data if loading fails
            sample_data = {
                'document': ["This is a sample legal document. It contains several sentences about a legal case. The plaintiff argued that the defendant breached their contract. The court found in favor of the plaintiff."],
                'summary': ["Court ruled for plaintiff in contract breach case."]
            }
            df = pd.DataFrame(sample_data)
            print("Using sample data for demonstration")
        else:
            # Analyze dataset and generate visualizations
            df = analyze_dataset(df)
            
            # Step 2: Preprocess the data
            print("\nStep 2: Preprocessing the data...")
            train_df, val_df, test_df = preprocess_data(df)
    except Exception as e:
        print(f"Error with dataset: {e}, using sample data")
        sample_data = {
            'document': ["This is a sample legal document. It contains several sentences about a legal case. The plaintiff argued that the defendant breached their contract. The court found in favor of the plaintiff."],
            'summary': ["Court ruled for plaintiff in contract breach case."]
        }
        df = pd.DataFrame(sample_data)
        train_df = df
    
    # Step 3: Initialize Sydney's T5 model (30% progress for now)
    print("\nStep 3: Initializing Sydney's T5 model...")
    try:
        t5_summarizer = T5LegalSummarizer()
        print("T5 model initialized successfully")
    except Exception as e:
        print(f"Error initializing T5 model: {e}")
        t5_summarizer = None
    
    # Step 4: Initialize Nicolas's BERTSUM model (30% progress for now)
    print("\nStep 4: Initializing Nicolas's BERTSUM model...")
    try:
        bertsum_extractor = BERTSumExtractor()
        print("BERTSUM model initialized successfully")
    except Exception as e:
        print(f"Error initializing BERTSUM model: {e}")
        bertsum_extractor = None
    
    # Step 5: Test with a small sample for demo purposes
    print("\nStep 5: Generating example summaries...")
    try:
        sample_doc = train_df['document'].iloc[0]
        sample_ref_summary = train_df['summary'].iloc[0]
    except:
        sample_doc = df['document'][0]
        sample_ref_summary = df['summary'][0]
    
    print("\nOriginal document sample (truncated):")
    print(sample_doc[:500] + "..." if len(sample_doc) > 500 else sample_doc)
    
    print("\nReference summary:")
    print(sample_ref_summary)
    
    # For demonstration purposes, we'll only pretend to use the models
    # (since they're not trained yet)
    print("\nT5 generated summary would appear here in the final project")
    print("\nBERTSUM extracted summary would appear here in the final project")
    
    print("\nProject setup completed. Next steps:")
    print("1. Fine-tune the T5 model with the prepared training data")
    print("2. Train the BERTSUM model for extractive summarization")
    print("3. Evaluate both models using ROUGE and BLEU metrics")
    print("4. Compare performance of abstractive vs extractive approaches")

if __name__ == "__main__":
    main()