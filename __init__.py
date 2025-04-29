# __init__.py at the root level
# This file makes the directory a Python package and enables relative imports

# DO NOT import modules here to avoid circular imports
# Instead, let the specific modules import what they need directly

# If you need to expose modules at the package level, uncomment these lines:
# from dataset.data_utils import load_data, preprocess_data, analyze_dataset
# from abstractive.t5_summarizer import T5LegalSummarizer
# from extractive.bertsum_extractor import BERTSumExtractor
# from utilities.evaluation import SummarizationEvaluator