# AI-Powered Legal Document Summarization

This project implements and compares two state-of-the-art approaches for legal document summarization:
1. **T5-based Abstractive Summarization** (by Sydney Ani)
2. **BERTSUM Extractive Summarization** (by Nicolas Osorio)

The project evaluates both approaches on legal documents and compares their performance using ROUGE and BLEU metrics.

## Project Structure

```
├── abstractive/
│   └── t5_summarizer.py       # T5-based abstractive summarization model
├── dataset/
│   └── data_utils.py          # Data loading and preprocessing utilities
├── extractive/
│   └── bertsum_extractor.py   # BERTSUM extractive summarization model
├── evaluation.py              # Evaluation metrics for summarization
├── main.py                    # Main script to run the project
├── outputs/                   # Output directory for models and results
└── README.md                  # This file
```

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install transformers torch nltk pandas numpy matplotlib seaborn scikit-learn rouge-score tqdm
```
3. Download the NLTK punkt tokenizer:
```python
import nltk
nltk.download('punkt')
```

## Evaluation Metrics

The project evaluates summarization quality using the following metrics:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
   - ROUGE-1: Overlap of unigrams between generated and reference summaries
   - ROUGE-2: Overlap of bigrams between generated and reference summaries
   - ROUGE-L: Longest Common Subsequence between generated and reference summaries

2. **BLEU (Bilingual Evaluation Understudy)**
   - Measures n-gram precision between generated and reference summaries
   - Includes scores for different n-gram sizes (BLEU-1, BLEU-2, BLEU-3, BLEU-4)

The evaluation results are saved in the output directory as a text file and visualized in a comparison chart.

### Examples

Training both models with 5 epochs and saving the results:
```bash
python main.py --mode both --epochs 5 --batch_size 8 --save_models
```

Training only the T5 model:
```bash
python main.py --mode t5 --epochs 3
```

Evaluating pre-trained models:
```bash
python main.py --eval_only
```

Running in interactive mode after training:
```bash
python main.py --epochs 1 --interactive
```

## Dataset

The project expects legal document data in the following folder structure:
```
├── UK-Abs/
│   └── case1/
│       ├── judgement/
│       │   └── document1.txt
│       └── summary/
│           └── document1.txt
├── IN-Ext/
│   └── ...
└── IN-Abs/
    └── ...
```

If the dataset is not available, the system will automatically use sample data for demonstration purposes.

## Model Descriptions

### T5 Abstractive Summarization

The T5 (Text-to-Text Transfer Transformer) model treats summarization as a text-to-text task, where the input is a document prefixed with "summarize: " and the output is the summary. Key features:

- Uses the T5 pre-trained model from Hugging Face
- Fine-tuned on legal documents
- Generates abstractive summaries that may contain novel phrases not in the original text
- Trained with teacher forcing for sequence-to-sequence learning

### BERTSUM Extractive Summarization

The BERTSUM model treats summarization as a sentence classification task, where each sentence is either included in the summary or not. Key features:

- Uses BERT for sentence encoding
- Adds a simple classification layer for sentence importance scoring
- Extracts the most important sentences based on their predicted scores
- Maintains the original wording of the document
- Uses ROUGE scores against reference summaries for training data preparation

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

This will:
1. Load and analyze the dataset
2. Preprocess the data
3. Initialize the summarization models
4. Train the models
5. Evaluate and compare the models
6. Display sample summaries
7. Save detailed evaluation results and model comparisons

### Command Line Arguments

The script accepts several command line arguments:

```bash
python main.py --mode both --epochs 3 --batch_size 8 --output_dir outputs --sample_data
```

- `--mode`: Choose which model to train/evaluate: `t5`, `bertsum`, or `both` (default: `both`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 8)
- `--t5_model`: T5 model name to use (default: 't5-base')
- `--bert_model`: BERT model name to use (default: 'bert-base-uncased')
- `--output_dir`: Directory to save models and results (default: 'outputs')
- `--sample_data`: Use sample data instead of loading the real dataset
- `--eval_only`: Skip training and only evaluate
- `--save_models`: Save trained models to disk
- `--interactive`: Run an interactive demo after training/evaluation