# AI-Powered Legal Document Summarization

This project implements and compares two state-of-the-art approaches for legal document summarization:

1. **T5-based Abstractive Summarization** (by Sydney Ani)
2. **BERTSUM Extractive Summarization** (by Nicolas Osorio)

The project evaluates both approaches on legal documents and compares their performance using ROUGE, BLEU, and BERTScore metrics.

## Project Structure

```
├── abstractive/
│   └── t5_summarizer.py       # T5-based abstractive summarization model
├── dataset/
│   └── data_utils.py          # Data loading and preprocessing utilities
├── extractive/
│   └── bertsum_extractor.py   # BERTSUM extractive summarization model
├── utilities/
│   └── evaluation.py          # Evaluation metrics for summarization
├── run_project.py             # Main script to run the project
├── outputs/                   # Output directory for models and results
├── install_dependencies.py    # Script for installing required packages
└── README.md                  # This file
```

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r utilities/requirements.txt
```
3. Download the NLTK tokenizers:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Evaluation Metrics

The project evaluates summarization quality using the following metrics:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
   - ROUGE-1: Overlap of unigrams between generated and reference summaries
   - ROUGE-2: Overlap of bigrams between generated and reference summaries
   - ROUGE-L: Longest Common Subsequence between generated and reference summaries

2. **BLEU (Bilingual Evaluation Understudy)**
   - Measures n-gram precision between generated and reference summaries
   - BLEU-1 and BLEU-4 are computed

3. **BERTScore**
   - Semantic similarity score using contextual embeddings from BERT

The evaluation results are saved in the `outputs/` directory and visualized in summary comparison tables and graphs.

### Examples

Training both models with 3 epochs:
```bash
python run_project.py --mode both --epochs 3 --save_models
```

Training only the T5 model:
```bash
python run_project.py --mode t5 --epochs 3
```

Evaluating pre-trained models:
```bash
python run_project.py --eval_only
```

Using a sample dataset for quick testing:
```bash
python run_project.py --sample_data
```

## Dataset

The project expects legal document data in the following structure:

```
├── UK-Abs/
│   ├── train-data/
│   │   ├── judgement/*.txt
│   │   └── summary/*.txt
├── IN-Ext/
│   └── ...
└── IN-Abs/
    └── ...
```

If these folders are missing, the project will default to using sample data for demonstration.

## Model Descriptions

### T5 Abstractive Summarization

- Utilizes Hugging Face's `t5-base` model
- Converts the summarization task into a sequence-to-sequence format
- Generates new phrases and reworded versions of input documents
- Fine-tuned using teacher forcing
- Particularly challenged by factual consistency in legal context

### BERTSUM Extractive Summarization

- Implements BERT with additional classification layers to rank sentence importance
- Retains original document language by selecting key sentences
- Preprocessing splits documents into sentence-level units
- Strong performance due to preservation of legal phrasing and terminology

## Usage

Run the project using:
```bash
python run_project.py
```

This executes the full pipeline:
1. Dataset analysis and visualization
2. Preprocessing and sentence tokenization
3. Training/fine-tuning models
4. Summarization generation
5. Evaluation using ROUGE, BLEU, BERTScore
6. Optional demo outputs printed in console

### Command Line Arguments

```bash
python run_project.py --mode both --epochs 3 --batch_size 8 --output_dir outputs --sample_data
```

- `--mode`: Model to run: `t5`, `bertsum`, or `both` (default: `both`)
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Batch size (default: 8)
- `--t5_model`: T5 variant (default: `t5-base`)
- `--bert_model`: BERT variant (default: `bert-base-uncased`)
- `--output_dir`: Path to save results (default: `outputs`)
- `--sample_data`: Use internal mini-dataset
- `--eval_only`: Skip training, run evaluation
- `--save_models`: Save model weights
- `--interactive`: Prompt for live summary testing after run

---

This project demonstrates a complete legal summarization pipeline combining preprocessing, abstractive and extractive modeling, and detailed evaluation. For more info, see the full report or contact the authors.
