# t5_summarizer.py
# Sydney Ani's T5-based Abstractive Summarization Model

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

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
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        print(f"T5 model loaded on {self.device}")
    
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
        Evaluates the model on the test set and computes ROUGE scores.
        """
        # This is a placeholder - in the actual project, you'll implement
        # ROUGE and BLEU evaluation here
        print("Model evaluation not yet implemented - will be added in the final project")