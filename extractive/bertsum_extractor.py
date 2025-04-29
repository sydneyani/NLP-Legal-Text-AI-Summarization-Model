# bertsum_extractor.py
# Nicolas Osorio's BERTSUM Extractive Summarization Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np

class SentenceDataset(Dataset):
    """Dataset for BERTSUM sentence encoding"""
    
    def __init__(self, sentences, labels=None):
        self.sentences = sentences
        self.labels = labels
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        item = {
            'sentence': self.sentences[idx],
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item


class BERTSumExtractor:
    """BERT-based model for extractive summarization of legal documents"""
    
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and BERT model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        
        # Define classifier for sentence importance prediction
        self.classifier = torch.nn.Linear(self.bert_model.config.hidden_size, 1)
        self.classifier.to(self.device)
        
        print(f"BERTSUM model initialized on {self.device}")
    
    def prepare_document(self, document):
        """
        Prepares a document by splitting it into sentences.
        """
        # Split document into sentences
        sentences = sent_tokenize(document)
        
        # Tokenize each sentence
        tokenized_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokenized_sentences.append({
                'sentence': sentence,
                'input_ids': tokens['input_ids'].to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device)
            })
        
        return tokenized_sentences
    
    def extract_sentence_embeddings(self, tokenized_sentences):
        """
        Extracts embeddings for each sentence using BERT.
        """
        self.bert_model.eval()
        sentence_embeddings = []
        
        with torch.no_grad():
            for sentence_data in tokenized_sentences:
                outputs = self.bert_model(
                    input_ids=sentence_data['input_ids'],
                    attention_mask=sentence_data['attention_mask']
                )
                
                # Use the [CLS] token embedding as the sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                sentence_embeddings.append(cls_embedding)
        
        return torch.cat(sentence_embeddings, dim=0)
    
    def prepare_training_data(self, df):
        """
        Prepares training data for BERTSUM by creating sentence-level labels.
        Uses ROUGE scores to identify important sentences.
        """
        from rouge_score import rouge_scorer
        
        print("Preparing BERTSUM training data...")
        all_documents = []
        all_sentence_labels = []
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            document = row['document']
            reference_summary = row['summary']
            
            # Split document into sentences
            sentences = sent_tokenize(document)
            
            if len(sentences) == 0:
                continue
                
            # Calculate ROUGE scores for each sentence against the reference summary
            rouge_scores = []
            for sentence in sentences:
                scores = scorer.score(reference_summary, sentence)
                # Use average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
                avg_score = (scores['rouge1'].fmeasure + 
                             scores['rouge2'].fmeasure + 
                             scores['rougeL'].fmeasure) / 3
                rouge_scores.append(avg_score)
            
            # Create binary labels: 1 for top N sentences, 0 for others
            # where N is based on the ratio of summary to document length
            summary_ratio = len(reference_summary) / len(document)
            num_to_select = max(1, int(len(sentences) * summary_ratio * 2))  # *2 to be more inclusive
            
            # Get indices of top sentences by ROUGE score
            top_indices = np.argsort(rouge_scores)[-num_to_select:]
            
            # Create binary labels
            labels = [1 if i in top_indices else 0 for i in range(len(sentences))]
            
            all_documents.append(sentences)
            all_sentence_labels.append(labels)
        
        return all_documents, all_sentence_labels
    
    def train(self, train_documents, train_labels, val_documents=None, val_labels=None, 
              epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Trains the BERTSUM model for extractive summarization.
        """
        # Flatten the nested lists
        all_sentences = [sentence for doc in train_documents for sentence in doc]
        all_labels = [label for doc_labels in train_labels for label in doc_labels]
        
        # Create dataset and dataloader
        train_dataset = SentenceDataset(all_sentences, all_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data if provided
        val_dataloader = None
        if val_documents and val_labels:
            val_sentences = [sentence for doc in val_documents for sentence in doc]
            val_labs = [label for doc_labels in val_labels for label in doc_labels]
            val_dataset = SentenceDataset(val_sentences, val_labs)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.bert_model.parameters(), 'lr': learning_rate / 10},
            {'params': self.classifier.parameters(), 'lr': learning_rate}
        ])
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        print(f"Starting BERTSUM training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.bert_model.train()
            self.classifier.train()
            train_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                sentences = batch['sentence']
                labels = torch.tensor(batch['label'], dtype=torch.float).to(self.device)
                
                # Tokenize sentences
                encoded = self.tokenizer(
                    sentences,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_embeddings).squeeze(-1)
                
                # Calculate loss
                loss = criterion(logits, labels)
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_dataloader:
                self.bert_model.eval()
                self.classifier.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        sentences = batch['sentence']
                        labels = torch.tensor(batch['label'], dtype=torch.float).to(self.device)
                        
                        # Tokenize sentences
                        encoded = self.tokenizer(
                            sentences,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                        
                        input_ids = encoded['input_ids'].to(self.device)
                        attention_mask = encoded['attention_mask'].to(self.device)
                        
                        # Forward pass
                        outputs = self.bert_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        cls_embeddings = outputs.last_hidden_state[:, 0, :]
                        logits = self.classifier(cls_embeddings).squeeze(-1)
                        
                        # Calculate loss
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_dataloader)
                    print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
        
        print("BERTSUM training completed!")
    
    def generate_extractive_summary(self, document, ratio=0.3):
        """
        Generates an extractive summary by selecting the most important sentences.
        """
        # Prepare document
        tokenized_sentences = self.prepare_document(document)
        
        if not tokenized_sentences:
            return "Unable to generate summary: No sentences found in document."
        
        # Extract sentence embeddings
        self.bert_model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            sentence_embeddings = []
            
            for sentence_data in tokenized_sentences:
                outputs = self.bert_model(
                    input_ids=sentence_data['input_ids'],
                    attention_mask=sentence_data['attention_mask']
                )
                
                # Use the [CLS] token embedding as the sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                sentence_embeddings.append(cls_embedding)
            
            # Stack embeddings and predict importance scores
            stacked_embeddings = torch.cat(sentence_embeddings, dim=0)
            scores = self.classifier(stacked_embeddings).squeeze(-1)
            
            # Select top sentences
            num_sentences = len(tokenized_sentences)
            num_to_select = max(1, int(num_sentences * ratio))
            
            # Get indices of top sentences
            top_indices = torch.topk(scores, min(num_to_select, num_sentences)).indices.cpu().numpy()
            
            # Sort indices to maintain original sentence order
            selected_indices = sorted(top_indices)
            
            # Combine selected sentences
            selected_sentences = [tokenized_sentences[i]['sentence'] for i in selected_indices]
            summary = ' '.join(selected_sentences)
        
        return summary
    
    def save_model(self, path):
        """
        Saves the trained model to the specified path.
        """
        torch.save({
            'bert_model_state_dict': self.bert_model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Loads a trained model from the specified path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify model name matches
        if checkpoint['model_name'] != self.model_name:
            print(f"Warning: Loaded model name ({checkpoint['model_name']}) doesn't match current model name ({self.model_name})")
        
        # Load model weights
        self.bert_model.load_state_dict(checkpoint['bert_model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.max_length = checkpoint['max_length']
        
        print(f"Model loaded from {path}")
    
    def evaluate(self, test_df):
        """
        Evaluates the BERTSUM model on test data.
        """
        from rouge_score import rouge_scorer
        import numpy as np
        
        print("Evaluating BERTSUM model...")
        
        # Initialize ROUGE scorer
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Store metrics
        rouge1_f = []
        rouge2_f = []
        rougeL_f = []
        
        # Generate summaries and evaluate
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            document = row['document']
            reference_summary = row['summary']
            
            # Generate extractive summary
            predicted_summary = self.generate_extractive_summary(document)
            
            # Calculate ROUGE scores
            scores = rouge_scorer_instance.score(reference_summary, predicted_summary)
            
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)
        
        # Calculate average scores
        results = {
            'rouge1_f': np.mean(rouge1_f),
            'rouge2_f': np.mean(rouge2_f),
            'rougeL_f': np.mean(rougeL_f)
        }
        
        print(f"ROUGE-1 F1: {results['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {results['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {results['rougeL_f']:.4f}")
        
        return results