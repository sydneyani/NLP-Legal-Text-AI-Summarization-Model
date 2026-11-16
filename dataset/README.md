# Legal Case Document Summarization Dataset

This folder contains the dataset used for the AI-Powered Legal Document Summarization project. The dataset was obtained from the paper "Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation" accepted at AACL-IJCNLP 2022.

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

`data_utils.py` is the loader used by `main.py`.

1. `load_data()` walks `UK-Abs/`, `IN-Ext/`, and `IN-Abs/`. It pairs files that share a filename and sit directly inside sibling `judgement/` and `summary/` directories. Nested summaries (`summary/full/A1`, `summary/segment-wise/`, and the UK test `summary/full/` tree) are not paired.
2. `preprocess_data()` collapses whitespace and drops documents of 50 characters or fewer and summaries of 10 characters or fewer.
3. The remaining rows are split 80% train, 10% validation, and 10% test with `random_state=42`. Fewer than 6 rows cannot form that split, so those rows are reused for all three sets. That is the path taken by the built-in sample in `main.py`.
4. `analyze_dataset()` prints character and sentence lengths and writes `dataset_analysis.png` at the repository root.

The statistics above are the analysis recorded with this project. The three published collections sum to 7,973 pairs, and the recorded analysis has 7,823. The raw files are not in the repository, so that difference cannot be reconstructed from this tree. Training and evaluation are documented in the [root README](../README.md).

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