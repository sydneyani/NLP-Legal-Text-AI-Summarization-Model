# bertsum_extractor.py
# Nicolas Osorio's BERTSUM Extractive Summarization Model

import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize

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
        This is a placeholder for now - in the actual project, you'll implement
        a method to determine which sentences from the original document
        should be included in the extractive summary.
        """
        print("BERTSUM training data preparation - to be implemented in the final project")
    
    def train(self, train_data, val_data, epochs=3, learning_rate=2e-5):
        """
        Trains the BERTSUM model for extractive summarization.
        This is a placeholder - in the actual project, you'll implement the training loop.
        """
        print("BERTSUM training loop - to be implemented in the final project")
    
    def generate_extractive_summary(self, document, ratio=0.3):
        """
        Generates an extractive summary by selecting the most important sentences.
        This is a simplified implementation for the initial project stage.
        """
        # Prepare document
        tokenized_sentences = self.prepare_document(document)
        
        # Extract sentence embeddings
        self.bert_model.eval()
        sentence_embeddings = self.extract_sentence_embeddings(tokenized_sentences)
        
        # Predict sentence importance scores
        self.classifier.eval()
        with torch.no_grad():
            scores = self.classifier(sentence_embeddings).squeeze(-1)
        
        # Select top sentences
        num_sentences = len(tokenized_sentences)
        num_to_select = max(1, int(num_sentences * ratio))
        
        selected_indices = torch.topk(scores, min(num_to_select, num_sentences)).indices.cpu().numpy()
        selected_indices = sorted(selected_indices)
        
        # Combine selected sentences
        selected_sentences = [tokenized_sentences[i]['sentence'] for i in selected_indices]
        summary = ' '.join(selected_sentences)
        
        return summary
    
    def evaluate(self, test_df):
        """
        Evaluates the BERTSUM model on test data.
        This is a placeholder - in the actual project, you'll implement
        evaluation using ROUGE and other metrics.
        """
        print("BERTSUM evaluation - to be implemented in the final project")