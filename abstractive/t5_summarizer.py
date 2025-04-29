# t5_summarizer.py
# Sydney Ani's T5-based Abstractive Summarization Model

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('t5_summarizer')

class LegalSummarizationDataset(Dataset):
    """Custom dataset class for T5 legal summarization"""
    
    def __init__(self, inputs, targets):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels = targets['input_ids']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class T5LegalSummarizer:
    """T5-based model for abstractive summarization of legal documents"""
    
    def __init__(self, model_name="t5-base", max_input_length=512, max_output_length=150):
        """
        Initialize the T5 summarizer with the specified model and parameters.
        
        Args:
            model_name (str): Name of the pre-trained T5 model to use
            max_input_length (int): Maximum length of input tokens
            max_output_length (int): Maximum length of output tokens
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check for and install dependencies if needed
        try:
            self._ensure_dependencies()
            
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            print(f"T5 model loaded on {self.device}")
        except Exception as e:
            print(f"Error initializing T5 model: {e}")
            raise
    
    def _ensure_dependencies(self):
        """Ensure all required dependencies are installed."""
        try:
            import sentencepiece
            logger.info("SentencePiece is already installed.")
        except ImportError:
            logger.warning("SentencePiece not found. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
                logger.info("SentencePiece installed successfully.")
                # Re-import to verify
                import sentencepiece
            except Exception as e:
                logger.error(f"Failed to install SentencePiece: {e}")
                raise ImportError(
                    "SentencePiece is required for T5Tokenizer but could not be installed. "
                    "Please install it manually with: pip install sentencepiece"
                )
    
    def prepare_data(self, train_df, val_df, test_df, batch_size=8):
        """
        Prepares the data for training the T5 model.
        """
        # Convert dataframes to HuggingFace datasets
        train_dataset = self._create_dataset(train_df)
        val_dataset = self._create_dataset(val_df)
        test_dataset = self._create_dataset(test_df)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size
        )
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def _create_dataset(self, df):
        """
        Creates a HuggingFace dataset from a pandas dataframe.
        """
        # Add prefix to documents for T5
        df['input_text'] = "summarize: " + df['document']
        
        # Tokenize inputs and targets
        inputs = self.tokenizer(
            df['input_text'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            df['summary'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = LegalSummarizationDataset(inputs, targets)
        return dataset
    
    def fine_tune(self, train_dataloader, val_dataloader, epochs=3, learning_rate=5e-5):
        """
        Fine-tunes the T5 model on legal summarization data.
        """
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        
        # Updated scheduler creation
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"Starting T5 fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
        print("T5 fine-tuning completed!")
    
    def generate_summary(self, text):
        """
        Generates a summary for a given legal text.
        """
        # Prepare input
        input_text = "summarize: " + text
        inputs = self.tokenizer(
            input_text, 
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and return summary
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary
    
    def evaluate(self, test_dataloader):
        """
        Evaluates the model on the test set and computes ROUGE and BLEU scores.
        
        Args:
            test_dataloader: DataLoader containing test data
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        from rouge_score import rouge_scorer
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize scorers
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1
        
        # Track metrics
        rouge1_p = []
        rouge1_r = []
        rouge1_f = []
        rouge2_p = []
        rouge2_r = []
        rouge2_f = []
        rougeL_p = []
        rougeL_r = []
        rougeL_f = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        
        # Generate predictions and calculate metrics
        self.model.eval()
        all_references = []
        all_predictions = []
        
        print("Evaluating T5 model on test set...")
        
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            # Generate summaries
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_output_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode generated summaries and reference summaries
            predicted_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            reference_summaries = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Add to collections
            all_predictions.extend(predicted_summaries)
            all_references.extend(reference_summaries)
            
            # Calculate metrics for each pair
            for ref, pred in zip(reference_summaries, predicted_summaries):
                # Skip empty strings
                if not ref.strip() or not pred.strip():
                    continue
                    
                # ROUGE scores
                rouge_scores = rouge_scorer_instance.score(ref, pred)
                
                rouge1_p.append(rouge_scores['rouge1'].precision)
                rouge1_r.append(rouge_scores['rouge1'].recall)
                rouge1_f.append(rouge_scores['rouge1'].fmeasure)
                
                rouge2_p.append(rouge_scores['rouge2'].precision)
                rouge2_r.append(rouge_scores['rouge2'].recall)
                rouge2_f.append(rouge_scores['rouge2'].fmeasure)
                
                rougeL_p.append(rouge_scores['rougeL'].precision)
                rougeL_r.append(rouge_scores['rougeL'].recall)
                rougeL_f.append(rouge_scores['rougeL'].fmeasure)
                
                # Tokenize for BLEU score
                ref_tokens = nltk.word_tokenize(ref)
                pred_tokens = nltk.word_tokenize(pred)
                
                # Calculate BLEU scores
                bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                bleu1_scores.append(bleu1)
                bleu2_scores.append(bleu2)
                bleu3_scores.append(bleu3)
                bleu4_scores.append(bleu4)
        
        # Calculate average scores
        results = {
            'rouge1_precision': np.mean(rouge1_p) if rouge1_p else 0,
            'rouge1_recall': np.mean(rouge1_r) if rouge1_r else 0,
            'rouge1_fmeasure': np.mean(rouge1_f) if rouge1_f else 0,
            'rouge2_precision': np.mean(rouge2_p) if rouge2_p else 0,
            'rouge2_recall': np.mean(rouge2_r) if rouge2_r else 0,
            'rouge2_fmeasure': np.mean(rouge2_f) if rouge2_f else 0,
            'rougeL_precision': np.mean(rougeL_p) if rougeL_p else 0,
            'rougeL_recall': np.mean(rougeL_r) if rougeL_r else 0,
            'rougeL_fmeasure': np.mean(rougeL_f) if rougeL_f else 0,
            'bleu_1': np.mean(bleu1_scores) if bleu1_scores else 0,
            'bleu_2': np.mean(bleu2_scores) if bleu2_scores else 0,
            'bleu_3': np.mean(bleu3_scores) if bleu3_scores else 0,
            'bleu_4': np.mean(bleu4_scores) if bleu4_scores else 0,
        }
        
        # Print summary of results
        print("\nT5 Evaluation Results:")
        print(f"ROUGE-1 F1: {results['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2 F1: {results['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L F1: {results['rougeL_fmeasure']:.4f}")
        print(f"BLEU-1: {results['bleu_1']:.4f}")
        print(f"BLEU-4: {results['bleu_4']:.4f}")
        
        return results, all_predictions