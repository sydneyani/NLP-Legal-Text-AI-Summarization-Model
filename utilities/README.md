# Evaluation utilities

`evaluation.py` defines `SummarizationEvaluator`. Given a test frame with `document` and `summary` columns, it generates a summary from either the T5 model or the BERT sentence ranker and scores the pair.

- **ROUGE-1, ROUGE-2, and ROUGE-L** via `rouge-score`, with Porter stemming. Precision, recall, and F1 are averaged over the test pairs.
- **BLEU-1 through BLEU-4** via NLTK `sentence_bleu` with smoothing method 1, then averaged.
- **BERTScore** (English, baseline-rescaled) when `bert-score` is installed.

The project overview, training commands, dataset notes, and the reported score table are in the root [README](../README.md). Install dependencies from the root [`requirements.txt`](../requirements.txt).
