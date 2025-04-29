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
import argparse
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure paths are correct
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import project modules
# Option 2: Direct imports from submodules
from dataset.data_utils import load_data, preprocess_data, analyze_dataset
from abstractive.t5_summarizer import T5LegalSummarizer
from extractive.bertsum_extractor import BERTSumExtractor
from utilities.evaluation import SummarizationEvaluator
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Download NLTK resources
nltk.download('punkt')

def create_sample_data():
    """Create sample data for demonstration purposes."""
    sample_data = {
        'document': [
            """This is a sample legal document. It contains several sentences about a legal case. 
            The plaintiff argued that the defendant breached their contract. The court found in favor of the plaintiff. 
            The damages were assessed at $50,000. The defendant was ordered to pay within 30 days. 
            The court also issued an injunction preventing similar actions in the future.""",
            """In the case of Smith v. Jones, the appellant sought to overturn the lower court's decision. 
            The appellant argued that the judge had erred in their interpretation of the contract clause. 
            The respondent maintained that the original interpretation was correct according to established precedent. 
            The court reviewed the evidence and upheld the lower court's decision. The appeal was dismissed with costs.""",
            """The Supreme Court granted certiorari to resolve a circuit split on the interpretation of the statute. 
            The petitioner argued that the language of the statute was ambiguous and should be interpreted in favor of the defendant. 
            The government contended that the legislative history clearly supported their broader interpretation. 
            After carefully reviewing the text and relevant precedents, the Court ruled 6-3 in favor of the government's interpretation. 
            Justice Roberts delivered the majority opinion, with Justices Thomas, Alito, and Gorsuch dissenting."""
        ],
        'summary': [
            "Court ruled for plaintiff in contract breach case, awarding $50,000 in damages and an injunction against the defendant.",
            "Appeal dismissed, lower court's interpretation of contract clause upheld with costs assigned to appellant.",
            "Supreme Court ruled 6-3 supporting government's interpretation of disputed statute, resolving circuit split."
        ]
    }
    return pd.DataFrame(sample_data)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Legal Document Summarization')
    parser.add_argument('--mode', type=str, default='both', choices=['t5', 'bertsum', 'both'],
                        help='Which model to train and evaluate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--t5_model', type=str, default='t5-base', help='T5 model name')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for models and results')
    parser.add_argument('--sample_data', action='store_true', help='Use sample data instead of loading real dataset')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--interactive', action='store_true', 
                      help='Run in interactive mode after training/evaluation')
    return parser.parse_args()

def train_t5_model(t5_summarizer, train_df, val_df, args):
    """Train T5 abstractive summarization model."""
    print("\nPreparing data for T5 model...")
    train_dataloader, val_dataloader, _ = t5_summarizer.prepare_data(
        train_df, val_df, val_df, batch_size=args.batch_size
    )
    
    print("\nFine-tuning T5 model...")
    t5_summarizer.fine_tune(
        train_dataloader, val_dataloader, epochs=args.epochs
    )
    
    if args.save_models:
        os.makedirs(args.output_dir, exist_ok=True)
        t5_path = os.path.join(args.output_dir, 't5_model.pt')
        torch.save(t5_summarizer.model.state_dict(), t5_path)
        print(f"T5 model saved to {t5_path}")
    
    return t5_summarizer

def train_bertsum_model(bertsum_extractor, train_df, val_df, args):
    """Train BERTSUM extractive summarization model."""
    print("\nPreparing data for BERTSUM model...")
    train_documents, train_labels = bertsum_extractor.prepare_training_data(train_df)
    val_documents, val_labels = bertsum_extractor.prepare_training_data(val_df)
    
    print("\nTraining BERTSUM model...")
    bertsum_extractor.train(
        train_documents, train_labels, val_documents, val_labels,
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    if args.save_models:
        os.makedirs(args.output_dir, exist_ok=True)
        bertsum_path = os.path.join(args.output_dir, 'bertsum_model.pt')
        bertsum_extractor.save_model(bertsum_path)
    
    return bertsum_extractor

def evaluate_models(t5_summarizer, bertsum_extractor, test_df, args):
    """Evaluate both models and compare performance."""
    print("\nStep 6: Evaluating models on test set...")
    
    # Initialize evaluator
    evaluator = SummarizationEvaluator()
    
    results = {}
    predictions = {}
    
    # Evaluate T5 model if available
    if t5_summarizer is not None and args.mode in ['t5', 'both']:
        print("\nEvaluating T5 Abstractive model...")
        t5_results, t5_predictions = evaluator.evaluate_model(t5_summarizer, test_df, extractive=False)
        results['t5'] = t5_results
        predictions['t5'] = t5_predictions
        
        # Print T5 results
        print("\nT5 Abstractive Results:")
        print(f"ROUGE-1 F1: {t5_results['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2 F1: {t5_results['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L F1: {t5_results['rougeL_fmeasure']:.4f}")
        print(f"BLEU-1: {t5_results['bleu_1']:.4f}")
        print(f"BLEU-4: {t5_results['bleu_4']:.4f}")
        print(f"BERTScore F1: {t5_results['bertscore_f1']:.4f}")
    
    # Evaluate BERTSUM model if available
    if bertsum_extractor is not None and args.mode in ['bertsum', 'both']:
        print("\nEvaluating BERTSUM Extractive model...")
        bertsum_results, bertsum_predictions = evaluator.evaluate_model(bertsum_extractor, test_df, extractive=True)
        results['bertsum'] = bertsum_results
        predictions['bertsum'] = bertsum_predictions
        
        # Print BERTSUM results
        print("\nBERTSUM Extractive Results:")
        print(f"ROUGE-1 F1: {bertsum_results['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2 F1: {bertsum_results['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L F1: {bertsum_results['rougeL_fmeasure']:.4f}")
        print(f"BLEU-1: {bertsum_results['bleu_1']:.4f}")
        print(f"BLEU-4: {bertsum_results['bleu_4']:.4f}")
        print(f"BERTScore F1: {bertsum_results['bertscore_f1']:.4f}")
    
    # Compare models if both were evaluated
    if 't5' in results and 'bertsum' in results:
        print("\nModel Comparison:")
        print("                  | T5 Abstractive | BERTSUM Extractive")
        print("------------------+----------------+-------------------")
        print(f"ROUGE-1 F1       | {results['t5']['rouge1_fmeasure']:.4f}         | {results['bertsum']['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2 F1       | {results['t5']['rouge2_fmeasure']:.4f}         | {results['bertsum']['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L F1       | {results['t5']['rougeL_fmeasure']:.4f}         | {results['bertsum']['rougeL_fmeasure']:.4f}")
        print(f"BLEU-1           | {results['t5']['bleu_1']:.4f}         | {results['bertsum']['bleu_1']:.4f}")
        print(f"BLEU-4           | {results['t5']['bleu_4']:.4f}         | {results['bertsum']['bleu_4']:.4f}")
        print(f"BERTScore F1     | {results['t5']['bertscore_f1']:.4f}         | {results['bertsum']['bertscore_f1']:.4f}")
        
        # Visualize comparison
        try:
            # Create bar chart comparing models
            # Updated to include BERTScore
            metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-1', 'BLEU-4', 'BERTScore']
            t5_scores = [results['t5']['rouge1_fmeasure'], results['t5']['rouge2_fmeasure'], 
                         results['t5']['rougeL_fmeasure'], results['t5']['bleu_1'], 
                         results['t5']['bleu_4'], results['t5']['bertscore_f1']]
            bertsum_scores = [results['bertsum']['rouge1_fmeasure'], results['bertsum']['rouge2_fmeasure'], 
                             results['bertsum']['rougeL_fmeasure'], results['bertsum']['bleu_1'], 
                             results['bertsum']['bleu_4'], results['bertsum']['bertscore_f1']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig = plt.figure(figsize=(12, 6))  # Slightly wider to accommodate more metrics
            ax = plt.subplot(111)
            
            rects1 = ax.bar(x - width/2, t5_scores, width, label='T5 Abstractive')
            rects2 = ax.bar(x + width/2, bertsum_scores, width, label='BERTSUM Extractive')
            
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
            print(f"\nModel comparison visualization saved to {os.path.join(args.output_dir, 'model_comparison.png')}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    # Save evaluation results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("Legal Document Summarization - Evaluation Results\n")
        f.write("=================================================\n\n")
        
        if 't5' in results:
            f.write("T5 Abstractive Model Results:\n")
            f.write("--------------------------\n")
            for metric, value in results['t5'].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
        
        if 'bertsum' in results:
            f.write("BERTSUM Extractive Model Results:\n")
            f.write("-------------------------------\n")
            for metric, value in results['bertsum'].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
        
        # Example predictions
        f.write("Example Predictions:\n")
        f.write("-------------------\n")
        
        for i in range(min(3, len(test_df))):
            f.write(f"Document {i+1}:\n")
            f.write(f"{test_df['document'].iloc[i][:300]}...\n\n")
            
            f.write(f"Reference Summary:\n")
            f.write(f"{test_df['summary'].iloc[i]}\n\n")
            
            if 't5' in predictions:
                f.write(f"T5 Generated Summary:\n")
                f.write(f"{predictions['t5'][i]}\n\n")
            
            if 'bertsum' in predictions:
                f.write(f"BERTSUM Generated Summary:\n")
                f.write(f"{predictions['bertsum'][i]}\n\n")
            
            f.write("-" * 80 + "\n\n")
    
    print(f"\nDetailed evaluation results saved to {results_file}")
    return results, predictions

def compare_example_outputs(t5_summarizer, bertsum_extractor, example_doc):
    """Generate and compare summaries for an example document."""
    print("\nExample Document:")
    print(example_doc[:300] + "..." if len(example_doc) > 300 else example_doc)
    
    results = []
    
    # Generate T5 summary if available
    if t5_summarizer is not None:
        try:
            t5_summary = t5_summarizer.generate_summary(example_doc)
            print("\nT5 Abstractive Summary:")
            print(t5_summary)
            results.append(("T5 Abstractive", t5_summary))
        except Exception as e:
            print(f"Error generating T5 summary: {e}")
    
    # Generate BERTSUM summary if available
    if bertsum_extractor is not None:
        try:
            bertsum_summary = bertsum_extractor.generate_extractive_summary(example_doc)
            print("\nBERTSUM Extractive Summary:")
            print(bertsum_summary)
            results.append(("BERTSUM Extractive", bertsum_summary))
        except Exception as e:
            print(f"Error generating BERTSUM summary: {e}")
    
    return results

def run_interactive_demo(t5_summarizer, bertsum_extractor):
    """Run an interactive demo allowing user to input custom legal text."""
    print("\n" + "="*80)
    print("         INTERACTIVE LEGAL DOCUMENT SUMMARIZATION DEMO")
    print("="*80)
    print("\nEnter a legal document to summarize (or 'q' to quit):")
    
    while True:
        # Get user input
        print("\nEnter legal text (or 'q' to quit):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'q':
                return
            if line.strip() == '' and lines:
                break
            lines.append(line)
        
        document = '\n'.join(lines)
        if not document.strip():
            continue
        
        # Generate summaries
        compare_example_outputs(t5_summarizer, bertsum_extractor, document)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and analyze the dataset
    print("Step 1: Loading and analyzing the dataset...")
    
    if args.sample_data:
        # Use sample data for demonstration
        df = create_sample_data()
        print("Using sample data for demonstration")
    else:
        # Try to load the actual dataset
        try:
            df = load_data()
            if df is None:
                # Fallback to sample data if loading fails
                df = create_sample_data()
                print("Dataset loading failed, using sample data instead")
        except Exception as e:
            print(f"Error loading dataset: {e}, using sample data")
            df = create_sample_data()
    
    # Analyze dataset and generate visualizations
    df = analyze_dataset(df)
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing the data...")
    train_df, val_df, test_df = preprocess_data(df)
    
    # Step 3: Initialize models
    t5_summarizer = None
    bertsum_extractor = None
    
    # Replace the existing Step 3a section in main.py with this code
    
    if args.mode in ['t5', 'both']:
        print("\nStep 3a: Initializing T5 Abstractive model...")
        try:
            # Check if sentencepiece is installed
            try:
                import sentencepiece
                print("SentencePiece library found.")
            except ImportError:
                print("Warning: SentencePiece library not found.")
                print("Installing SentencePiece...")
                import subprocess
                subprocess.check_call(["pip", "install", "sentencepiece"])
                print("SentencePiece installed. Continuing...")
                
            # Try to initialize the T5 model
            t5_summarizer = T5LegalSummarizer(model_name=args.t5_model)
            print("T5 model initialized successfully")
        except ImportError as e:
            print(f"Error initializing T5 model: {e}")
            print("Please install the required libraries with:")
            print("pip install sentencepiece protobuf")
            print("Then restart your Python runtime to ensure the libraries are properly loaded.")
            t5_summarizer = None
        except Exception as e:
            print(f"Error initializing T5 model: {e}")
            t5_summarizer = None
    
    # Step 4: Training models (if not eval_only)
    if not args.eval_only:
        if t5_summarizer is not None and args.mode in ['t5', 'both']:
            print("\nStep 4a: Training T5 Abstractive model...")
            t5_summarizer = train_t5_model(t5_summarizer, train_df, val_df, args)
        
        if bertsum_extractor is not None and args.mode in ['bertsum', 'both']:
            print("\nStep 4b: Training BERTSUM Extractive model...")
            bertsum_extractor = train_bertsum_model(bertsum_extractor, train_df, val_df, args)
    
    # Step 5: Evaluate models
    results, predictions = evaluate_models(t5_summarizer, bertsum_extractor, test_df, args)
    
    # Step 6: Display sample summaries
    print("\nStep 6: Sample Summarization Examples")
    
    # Pick a random document from test set
    import random
    sample_idx = random.randint(0, len(test_df) - 1)
    sample_doc = test_df['document'].iloc[sample_idx]
    sample_ref_summary = test_df['summary'].iloc[sample_idx]
    
    print("\nOriginal document sample (truncated):")
    print(sample_doc[:300] + "..." if len(sample_doc) > 300 else sample_doc)
    
    print("\nReference summary:")
    print(sample_ref_summary)
    
    # Generate sample summaries with trained models
    if t5_summarizer is not None and args.mode in ['t5', 'both']:
        print("\nT5 Abstractive summary:")
        try:
            t5_sample_summary = t5_summarizer.generate_summary(sample_doc)
            print(t5_sample_summary)
        except Exception as e:
            print(f"Error generating T5 summary: {e}")
    
    if bertsum_extractor is not None and args.mode in ['bertsum', 'both']:
        print("\nBERTSUM Extractive summary:")
        try:
            bertsum_sample_summary = bertsum_extractor.generate_extractive_summary(sample_doc)
            print(bertsum_sample_summary)
        except Exception as e:
            print(f"Error generating BERTSUM summary: {e}")
    
    print("\nProject completed successfully!")
    print(f"All outputs saved to {args.output_dir} directory")

if __name__ == "__main__":
    # Parse command line arguments for interactive mode
    parser = argparse.ArgumentParser(description='Legal Document Summarization')
    parser.add_argument('--interactive', action='store_true', 
                      help='Run in interactive mode after training/evaluation')
    args, unknown = parser.parse_known_args()
    
    # Run main training/evaluation process
    main()
    
    # Run interactive demo if requested
    if args.interactive:
        # Initialize models again for demo
        t5_summarizer = T5LegalSummarizer()
        bertsum_extractor = BERTSumExtractor()
        
        # Parse args again to get output_dir
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_dir', type=str, default='outputs')
        args, _ = parser.parse_known_args()
        
        # Try to load saved models if available
        try:
            t5_path = os.path.join(args.output_dir, 't5_model.pt')
            if os.path.exists(t5_path):
                t5_summarizer.model.load_state_dict(torch.load(t5_path, map_location=t5_summarizer.device))
                print(f"Loaded saved T5 model from {t5_path}")
        except Exception as e:
            print(f"Error loading T5 model: {e}")
        
        try:
            bertsum_path = os.path.join(args.output_dir, 'bertsum_model.pt')
            if os.path.exists(bertsum_path):
                bertsum_extractor.load_model(bertsum_path)
                print(f"Loaded saved BERTSUM model from {bertsum_path}")
        except Exception as e:
            print(f"Error loading BERTSUM model: {e}")
        
        run_interactive_demo(t5_summarizer, bertsum_extractor)