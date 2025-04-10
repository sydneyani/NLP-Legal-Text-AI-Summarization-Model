# data_utils.py
# Data utilities for Legal Document Summarization project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize

def load_data():
    """
    Loads the legal case dataset from text files in the nested folder structure.
    """
    # Define the base folders
    base_folders = ['UK-Abs', 'IN-Ext', 'IN-Abs']
    
    # Initialize lists to store documents and summaries
    documents = []
    summaries = []
    
    # Track if we found any data
    found_data = False
    processed_files = 0
    
    # Process each base folder
    for base_folder in base_folders:
        if not os.path.exists(base_folder):
            print(f"Folder not found: {base_folder}")
            continue
            
        print(f"Processing folder: {base_folder}")
        found_data = True
        
        # Get all subfolders
        for root, dirs, files in os.walk(base_folder):
            # Look for 'judgement' and 'summary' folders
            judgement_folder = None
            summary_folder = None
            
            for d in dirs:
                if d.lower() == 'judgement':
                    judgement_folder = os.path.join(root, d)
                elif d.lower() == 'summary':
                    summary_folder = os.path.join(root, d)
            
            # If we have both judgement and summary folders in the same parent
            if judgement_folder and summary_folder:
                # Get all files in judgement folder
                judgement_files = [f for f in os.listdir(judgement_folder) 
                                  if os.path.isfile(os.path.join(judgement_folder, f))]
                
                # Get all files in summary folder
                summary_files = [f for f in os.listdir(summary_folder) 
                               if os.path.isfile(os.path.join(summary_folder, f))]
                
                # Try to match files between folders
                for j_file in judgement_files:
                    # Try to find matching summary file with same name
                    if j_file in summary_files:
                        # Read the judgement file
                        try:
                            with open(os.path.join(judgement_folder, j_file), 'r', encoding='utf-8') as f:
                                judgement_text = f.read().strip()
                            
                            # Read the summary file
                            with open(os.path.join(summary_folder, j_file), 'r', encoding='utf-8') as f:
                                summary_text = f.read().strip()
                            
                            # Add to our dataset
                            documents.append(judgement_text)
                            summaries.append(summary_text)
                            processed_files += 1
                            
                            if processed_files % 10 == 0:
                                print(f"Processed {processed_files} document-summary pairs")
                                
                        except Exception as e:
                            print(f"Error reading files {j_file}: {e}")
    
    if not found_data:
        print("Could not find any of the data folders (UK-Abs, IN-Ext, IN-Abs)")
        return None
    
    if not documents or not summaries:
        print("Could not find any matching document-summary pairs")
        print("Please ensure the folders contain 'judgement' and 'summary' subfolders with matching files")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'document': documents,
        'summary': summaries
    })
    
    print(f"Dataset created with {len(df)} document-summary pairs")
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset by cleaning text and splitting into train/val/test sets.
    """
    # Basic cleaning
    df['document'] = df['document'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    df['summary'] = df['summary'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    # Remove any rows with empty documents or summaries
    df = df.dropna(subset=['document', 'summary'])
    df = df[df['document'].str.len() > 50]  # Remove very short documents
    df = df[df['summary'].str.len() > 10]   # Remove very short summaries
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df

def analyze_dataset(df):
    """
    Performs basic analysis on the dataset.
    """
    # Document and summary lengths
    df['doc_length'] = df['document'].apply(len)
    df['summary_length'] = df['summary'].apply(len)
    df['doc_sentences'] = df['document'].apply(lambda x: len(sent_tokenize(x)))
    df['summary_sentences'] = df['summary'].apply(lambda x: len(sent_tokenize(x)))
    
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Average document length (chars): {df['doc_length'].mean():.2f}")
    print(f"Average summary length (chars): {df['summary_length'].mean():.2f}")
    print(f"Average document sentences: {df['doc_sentences'].mean():.2f}")
    print(f"Average summary sentences: {df['summary_sentences'].mean():.2f}")
    print(f"Compression ratio (summary/document): {(df['summary_length'] / df['doc_length']).mean():.4f}")
    
    # Create distribution plots
    try:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(df['doc_length'], bins=50, ax=axs[0, 0])
        axs[0, 0].set_title('Document Length Distribution')
        axs[0, 0].set_xlabel('Character Count')
        
        sns.histplot(df['summary_length'], bins=50, ax=axs[0, 1])
        axs[0, 1].set_title('Summary Length Distribution')
        axs[0, 1].set_xlabel('Character Count')
        
        sns.histplot(df['doc_sentences'], bins=30, ax=axs[1, 0])
        axs[1, 0].set_title('Document Sentence Count Distribution')
        axs[1, 0].set_xlabel('Number of Sentences')
        
        sns.histplot(df['summary_sentences'], bins=20, ax=axs[1, 1])
        axs[1, 1].set_title('Summary Sentence Count Distribution')
        axs[1, 1].set_xlabel('Number of Sentences')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png')
        plt.close()
        print("Dataset analysis plots saved to 'dataset_analysis.png'")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return df