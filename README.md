# AI-Powered Legal Document Summarization

## Team Members
- Sydney Ani
- Nicolas Osorio

## Introduction
This repository contains our implementation of AI-powered legal document summarization techniques as part of the EDS 6397 Natural Language Processing course project. Our goal is to develop an effective system that can automatically generate concise and informative summaries of legal case documents.

We explore both extractive and abstractive summarization approaches:
- **Extractive Summarization (Nicolas Osorio)**: Using BERTSUM to identify and extract the most important sentences from legal documents.
- **Abstractive Summarization (Sydney Ani)**: Fine-tuning T5 transformer models to generate human-like summaries that may include novel sentences not found in the original text.

## Dataset
We leverage the legal case summarization dataset from the paper "Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation" accepted at AACL-IJCNLP 2022.

The dataset includes:
- **IN-Abs**: Indian Supreme Court case documents & their abstractive summaries
- **IN-Ext**: Indian Supreme Court case documents & their extractive summaries
- **UK-Abs**: United Kingdom Supreme Court case documents & their abstractive summaries

For detailed information about the dataset structure and statistics, please see the [dataset/README.md](dataset/README.md) file.

## Key Features
- Data processing pipeline for complex nested folder structure
- Custom dataset classes for legal documents
- T5-based abstractive summarization implementation
- BERTSUM-based extractive summarization implementation
- Evaluation framework using ROUGE and BLEU metrics

## Current Progress
- Successfully loaded and processed 7,823 document-summary pairs
- Implemented data preprocessing and analysis
- Set up model architectures for both approaches:
  - T5 model for abstractive summarization
  - BERTSUM model for extractive summarization
- Established training and evaluation pipelines

## Dataset Statistics
- **Total samples**: 7,823 document-summary pairs
- **Average document length**: 30,339.68 characters (~190 sentences)
- **Average summary length**: 4,996.06 characters (~33 sentences)
- **Compression ratio**: 22.52% (summary/document length)

## Project Structure
```
NLP-Legal-Text-AI-Summarization-Model/
├── abstractive/           # Sydney's T5-based abstractive summarization
├── dataset/               # Dataset folder and utilities
├── extractive/            # Nicolas's BERTSUM extractive summarization
├── utilities/             # Shared utility scripts
├── evaluation.py          # Evaluation metrics and functions
├── main.py                # Main execution script
└── README.md              # This file
```

## Future Work
- Complete model training for both approaches
- Implement comprehensive evaluation using ROUGE and BLEU
- Optimize hyperparameters for legal document domain
- Compare effectiveness of extractive vs. abstractive approaches

## Citation
If you use the dataset in your work, please cite the original papers:

```bibtex
@inproceedings{shukla2022,
  title={Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation},
  author={Shukla, Abhay and Bhattacharya, Paheli and Poddar, Soham and Mukherjee, Rajdeep and Ghosh, Kripabandhu and Goyal, Pawan and Ghosh, Saptarshi},
  booktitle={The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
  year={2022}
}
```

## Acknowledgements
We would like to thank the authors of the original dataset for making it publicly available.