# Legal Case Document Summarization Dataset

This folder documents the corpus used by the legal summarization project. The text files themselves are not stored in the repository. The dataset was released with the paper "Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation" (AACL-IJCNLP 2022). Setup, training, and reported scores are in the project [README](../README.md).

## Dataset Overview

The original dataset repository can be downloaded from [Zenodo](https://zenodo.org/record/7152317#.Yz6mJ9JByC0).

It consists of 3 main datasets:

1. **IN-Abs**: Indian Supreme Court case documents & their abstractive summaries
   - Source: [LII of India](http://www.liiofindia.org/in/cases/cen/INSC/)
   - 7,130 full case documents with corresponding abstractive summaries
   - 7,030 pairs for training and 100 pairs for testing

2. **IN-Ext**: Indian Supreme Court case documents & their extractive summaries
   - 50 case documents with extractive summaries written by two law experts (A1, A2)
   - Each summary is available in two formats: full and segment-wise

3. **UK-Abs**: United Kingdom Supreme Court case documents & their abstractive summaries
   - Source: [UK Supreme Court](https://www.supremecourt.uk/decided-cases/)
   - 793 full case documents with corresponding abstractive summaries
   - 693 pairs for training and 100 pairs for testing

## Directory Structure

```
.
├── IN-Abs/                      # Indian Supreme Court - Abstractive summaries
│   ├── train-data/              # 7,030 document-summary pairs for training
│   │   ├── judgement/           # Original legal documents
│   │   ├── summary/             # Abstractive summaries
│   │   └── stats-IN-train.txt   # Statistics file
│   └── test-data/               # 100 document-summary pairs for testing
│       ├── judgement/
│       ├── summary/
│       └── stats-IN-test.txt
│
├── IN-Ext/                      # Indian Supreme Court - Extractive summaries
│   ├── judgement/               # 50 original legal documents
│   ├── summary/
│   │   ├── full/                # Complete summaries
│   │   │   ├── A1/              # First law expert
│   │   │   └── A2/              # Second law expert
│   │   └── segment-wise/        # Segmented summaries (analysis, argument, facts, etc.)
│   │       ├── A1/
│   │       └── A2/
│   └── IN-EXT-length.txt        # Statistics file
│
└── UK-Abs/                      # UK Supreme Court - Abstractive summaries
    ├── train-data/              # 693 document-summary pairs for training
    │   ├── judgement/
    │   ├── summary/
    │   └── stats-UK-train.txt
    └── test-data/               # 100 document-summary pairs for testing
        ├── judgement/
        ├── summary/
        │   ├── full/
        │   └── segment-wise/    # Segments: background, judgement, reasons
        └── stats-UK-test.txt
```

## Dataset Statistics

From our analysis of the combined dataset:

- **Total samples**: 7,823 document-summary pairs
- **Average document length**: 30,339.68 characters (approximately 190.04 sentences)
- **Average summary length**: 4,996.06 characters (approximately 32.91 sentences)
- **Compression ratio**: 0.2252 (summary length / document length)

## Usage in this repository

`data_utils.py` walks `UK-Abs`, `IN-Abs`, and `IN-Ext`. It keeps a pair when a `judgement/` directory and a `summary/` directory share a parent and contain the same filename. Nested summary folders such as `summary/full/A1/` sit below that match, so the two-expert and segment-wise files load after they are placed in a flat `summary/` directory next to `judgement/`.

Preprocessing collapses whitespace and drops a pair when the document is shorter than 50 characters or the summary is shorter than 10 characters. The remaining rows are split 80/10/10 with `random_state=42`. That split pools every loaded pair. It does not keep the official IN-Abs and UK-Abs test sets of 100 cases.

Download the corpus from [Zenodo record 7152317](https://zenodo.org/records/7152317) and place the three top-level folders at the repository root. Those folders are listed in `.gitignore`.

## Citation

If using this dataset, please cite the original paper:

```bibtex
@inproceedings{bhattacharya2021,
  title={Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation},
  author={Shukla, Abhay and Bhattacharya, Paheli and Poddar, Soham and Mukherjee, Rajdeep and Ghosh, Kripabandhu and Goyal, Pawan and Ghosh, Saptarshi},
  booktitle={The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
  year={2022}
}
```

*Note: The raw dataset files are not included in this repository due to size constraints. The data processing pipeline code is provided instead.*