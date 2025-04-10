# evaluation.py
# Evaluation utilities for legal document summarization

import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary NLTK resources
nltk.download('punkt')

class SummarizationEvaluator:
    """Class for evaluating summarization models using ROUGE and BLEU metrics"""
    
    def __init__(self):
        # Initialize ROUGE scorer with different variants
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
    
    def compute_rouge_scores(self, references, predictions):
        """
        Compute ROUGE scores for a list of reference and predicted summaries.
        
        Args:
            references (list): List of reference (human) summaries
            predictions (list): List of predicted (model) summaries
            
        Returns:
            dict: Dictionary containing average ROUGE scores
        """
        scores = {
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge1_fmeasure': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rouge2_fmeasure': [],
            'rougeL_precision': [],
            'rougeL_recall': [],
            'rougeL_fmeasure': []
        }
        
        for ref, pred in zip(references, predictions):
            # Calculate ROUGE scores for each summary pair
            rouge_scores = self.rouge_scorer.score(ref, pred)
            
            # Add scores to lists
            scores['rouge1_precision'].append(rouge_scores['rouge1'].precision)
            scores['rouge1_recall'].append(rouge_scores['rouge1'].recall)
            scores['rouge1_fmeasure'].append(rouge_scores['rouge1'].fmeasure)
            
            scores['rouge2_precision'].append(rouge_scores['rouge2'].precision)
            scores['rouge2_recall'].append(rouge_scores['rouge2'].recall)
            scores['rouge2_fmeasure'].append(rouge_scores['rouge2'].fmeasure)
            
            scores['rougeL_precision'].append(rouge_scores['rougeL'].precision)
            scores['rougeL_recall'].append(rouge_scores['rougeL'].recall)
            scores['rougeL_fmeasure'].append(rouge_scores['rougeL'].fmeasure)
        
        # Calculate averages
        avg_scores = {key: np.mean(values) for key, values in scores.items()}
        
        return avg_scores
    
    def compute_bleu_scores(self, references, predictions):
        """
        Compute BLEU scores for a list of reference and predicted summaries.
        
        Args:
            references (list): List of reference (human) summaries
            predictions (list): List of predicted (model) summaries
            
        Returns:
            dict: Dictionary containing average BLEU scores
        """
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for ref, pred in zip(references, predictions):
            # Tokenize reference and prediction
            ref_tokens = nltk.word_tokenize(ref)
            pred_tokens = nltk.word_tokenize(pred)
            
            # Calculate BLEU scores with different n-gram weights
            bleu_1 = sentence_bleu([ref_tokens], pred_tokens, 
                                  weights=(1, 0, 0, 0), 
                                  smoothing_function=self.smoothing)
            bleu_2 = sentence_bleu([ref_tokens], pred_tokens, 
                                  weights=(0.5, 0.5, 0, 0), 
                                  smoothing_function=self.smoothing)
            bleu_3 = sentence_bleu([ref_tokens], pred_tokens, 
                                  weights=(0.33, 0.33, 0.33, 0), 
                                  smoothing_function=self.smoothing)
            bleu_4 = sentence_bleu([ref_tokens], pred_tokens, 
                                  weights=(0.25, 0.25, 0.25, 0.25), 
                                  smoothing_function=self.smoothing)
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
        
        # Calculate averages
        avg_scores = {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_2': np.mean(bleu_2_scores),
            'bleu_3': np.mean(bleu_3_scores),
            'bleu_4': np.mean(bleu_4_scores)
        }
        
        return avg_scores
    
    def evaluate_model(self, model, test_df, extractive=False):
        """
        Evaluate a summarization model using ROUGE and BLEU metrics.
        
        Args:
            model: The summarization model (T5 or BERTSUM)
            test_df: DataFrame containing test data
            extractive: Boolean indicating if the model is extractive
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        references = test_df['summary'].tolist()
        predictions = []
        
        for _, row in test_df.iterrows():
            if extractive:
                # For BERTSUM
                pred_summary = model.generate_extractive_summary(row['document'])
            else:
                # For T5
                pred_summary = model.generate_summary(row['document'])
            
            predictions.append(pred_summary)
        
        # Compute metrics
        rouge_scores = self.compute_rouge_scores(references, predictions)
        bleu_scores = self.compute_bleu_scores(references, predictions)
        
        # Combine metrics
        results = {**rouge_scores, **bleu_scores}
        
        return results, predictions