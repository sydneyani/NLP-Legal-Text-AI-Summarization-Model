# evaluation.py
# Evaluation utilities for legal document summarization

import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import traceback
import logging
import torch
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SummarizationEvaluator:
    """Class for evaluating summarization models using ROUGE, BLEU, and BERTScore metrics"""
    
    def __init__(self):
        # Initialize ROUGE scorer with different variants
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
        # Initialize BERTScore if available
        self.bert_score_available = False
        try:
            self._ensure_bertscore()
            
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            self.bert_score_available = True
            logger.info("BERTScore initialized successfully")
        except ImportError:
            logger.warning("bert-score not found. Install with 'pip install bert-score' to use BERTScore.")
        except Exception as e:
            logger.warning(f"BERTScore initialization failed: {e}")
        
        logger.info("SummarizationEvaluator initialized")
    
    def _ensure_bertscore(self):
        """Ensure BERTScore is installed."""
        try:
            import bert_score
            logger.info("BERTScore already installed.")
        except ImportError:
            logger.warning("BERTScore not found. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score>=0.3.12"])
                logger.info("BERTScore installed successfully.")
                # Try to import to verify
                import bert_score
            except Exception as e:
                logger.error(f"Failed to install BERTScore: {e}")
                raise ImportError(
                    "BERTScore could not be installed automatically. "
                    "Please install it manually with: pip install bert-score"
                )
    
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
        
        logger.info(f"Computing ROUGE scores for {len(references)} summary pairs")
        for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Computing ROUGE scores"):
            # Handle empty strings
            if not ref.strip() or not pred.strip():
                logger.warning("Empty reference or prediction encountered, skipping")
                continue
                
            try:
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
            except Exception as e:
                logger.error(f"Error computing ROUGE scores: {e}")
                logger.error(f"Reference: {ref[:100]}...")
                logger.error(f"Prediction: {pred[:100]}...")
                logger.error(traceback.format_exc())
        
        # Calculate averages
        avg_scores = {}
        for key, values in scores.items():
            if values:
                avg_scores[key] = np.mean(values)
            else:
                avg_scores[key] = 0.0
                logger.warning(f"No valid scores for {key}")
        
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
        
        logger.info(f"Computing BLEU scores for {len(references)} summary pairs")
        for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Computing BLEU scores"):
            # Handle empty strings
            if not ref.strip() or not pred.strip():
                logger.warning("Empty reference or prediction encountered, skipping")
                continue
                
            try:
                # Tokenize reference and prediction
                ref_tokens = nltk.word_tokenize(ref)
                pred_tokens = nltk.word_tokenize(pred)
                
                # Skip if tokens are empty
                if not ref_tokens or not pred_tokens:
                    logger.warning("Empty tokens encountered, skipping")
                    continue
                
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
            except Exception as e:
                logger.error(f"Error computing BLEU scores: {e}")
                logger.error(f"Reference: {ref[:100]}...")
                logger.error(f"Prediction: {pred[:100]}...")
                logger.error(traceback.format_exc())
        
        # Calculate averages
        avg_scores = {
            'bleu_1': np.mean(bleu_1_scores) if bleu_1_scores else 0.0,
            'bleu_2': np.mean(bleu_2_scores) if bleu_2_scores else 0.0,
            'bleu_3': np.mean(bleu_3_scores) if bleu_3_scores else 0.0,
            'bleu_4': np.mean(bleu_4_scores) if bleu_4_scores else 0.0
        }
        
        return avg_scores
    
    def compute_bert_scores(self, references, predictions):
        """
        Compute BERTScore for a list of reference and predicted summaries.
        
        Args:
            references (list): List of reference (human) summaries
            predictions (list): List of predicted (model) summaries
            
        Returns:
            dict: Dictionary containing average BERTScores (P, R, F1)
        """
        # Try to install BERTScore again if not available
        if not self.bert_score_available:
            try:
                self._ensure_bertscore()
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                self.bert_score_available = True
                logger.info("BERTScore initialized successfully on second attempt")
            except Exception as e:
                logger.warning(f"BERTScore still unavailable: {e}")
                return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        # Filter out empty strings
        valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                       if ref.strip() and pred.strip()]
        
        if not valid_pairs:
            logger.warning("No valid reference-prediction pairs for BERTScore")
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        valid_refs, valid_preds = zip(*valid_pairs)
        
        try:
            logger.info(f"Computing BERTScore for {len(valid_refs)} summary pairs")
            P, R, F1 = self.bert_scorer.score(valid_preds, valid_refs)
            
            # Convert from torch tensors to numpy
            P = P.cpu().numpy()
            R = R.cpu().numpy()
            F1 = F1.cpu().numpy()
            
            return {
                'bertscore_precision': float(np.mean(P)),
                'bertscore_recall': float(np.mean(R)),
                'bertscore_f1': float(np.mean(F1))
            }
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            logger.error(traceback.format_exc())
            
            # Fall back to zero scores but make the error visible
            print(f"\nWARNING: BERTScore calculation failed: {e}")
            print("Continuing with evaluation using only ROUGE and BLEU metrics.")
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
    
    def evaluate_model(self, model, test_df, extractive=False):
        """
        Evaluate a summarization model using ROUGE, BLEU, and BERTScore metrics.
        
        Args:
            model: The summarization model (T5 or BERTSUM)
            test_df: DataFrame containing test data
            extractive: Boolean indicating if the model is extractive
            
        Returns:
            dict: Dictionary containing evaluation results
            list: List of predicted summaries
        """
        references = test_df['summary'].tolist()
        predictions = []
        
        model_type = "extractive" if extractive else "abstractive"
        logger.info(f"Evaluating {model_type} model on {len(test_df)} test examples")
        
        # Generate predictions
        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Generating {model_type} summaries"):
            try:
                if extractive:
                    # For BERTSUM
                    pred_summary = model.generate_extractive_summary(row['document'])
                else:
                    # For T5
                    pred_summary = model.generate_summary(row['document'])
                
                predictions.append(pred_summary)
            except Exception as e:
                logger.error(f"Error generating summary for example {i}: {e}")
                logger.error(traceback.format_exc())
                # Add empty prediction to maintain alignment with references
                predictions.append("")
        
        # Compute metrics
        rouge_scores = self.compute_rouge_scores(references, predictions)
        bleu_scores = self.compute_bleu_scores(references, predictions)
        bert_scores = self.compute_bert_scores(references, predictions)
        
        # Combine metrics
        results = {**rouge_scores, **bleu_scores, **bert_scores}
        
        # Print summary of results
        logger.info(f"{model_type.capitalize()} Model Evaluation Results:")
        logger.info(f"ROUGE-1 F1: {results['rouge1_fmeasure']:.4f}")
        logger.info(f"ROUGE-2 F1: {results['rouge2_fmeasure']:.4f}")
        logger.info(f"ROUGE-L F1: {results['rougeL_fmeasure']:.4f}")
        logger.info(f"BLEU-1: {results['bleu_1']:.4f}")
        logger.info(f"BLEU-4: {results['bleu_4']:.4f}")
        logger.info(f"BERTScore F1: {results['bertscore_f1']:.4f}")
        
        return results, predictions
    
    def compare_models(self, t5_results, bertsum_results):
        """
        Compare the performance of T5 and BERTSUM models.
        
        Args:
            t5_results (dict): Results from T5 model evaluation
            bertsum_results (dict): Results from BERTSUM model evaluation
            
        Returns:
            dict: Comparison results showing which model performed better on each metric
        """
        comparison = {}
        
        for metric in t5_results.keys():
            if metric not in bertsum_results:
                continue
                
            t5_score = t5_results[metric]
            bertsum_score = bertsum_results[metric]
            
            if t5_score > bertsum_score:
                winner = "T5"
                diff = t5_score - bertsum_score
            elif bertsum_score > t5_score:
                winner = "BERTSUM"
                diff = bertsum_score - t5_score
            else:
                winner = "Tie"
                diff = 0.0
            
            comparison[metric] = {
                "winner": winner,
                "difference": diff,
                "t5_score": t5_score,
                "bertsum_score": bertsum_score
            }
        
        # Log comparison summary
        logger.info("Model Comparison Summary:")
        for metric, result in comparison.items():
            if "fmeasure" in metric or metric.startswith("bleu") or "bertscore" in metric:
                logger.info(f"{metric}: {result['winner']} wins by {result['difference']:.4f}")
        
        return comparison