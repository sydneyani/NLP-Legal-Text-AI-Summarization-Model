# Evaluation utilities

`evaluation.py` defines `SummarizationEvaluator`. Given a reference summary and a model summary, it computes:

- ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1, with stemming
- Sentence BLEU-1 through BLEU-4, with NLTK smoothing method 1
- BERTScore precision, recall, and F1, with baseline rescaling

`main.py` calls `evaluate_model` on the test split and writes `outputs/evaluation_results.txt`.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` in this folder lists the same packages.
