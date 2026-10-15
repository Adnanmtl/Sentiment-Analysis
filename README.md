# Sentiment Analysis

A comprehensive natural language processing (NLP) project for sentiment classification of customer reviews. It compares traditional machine learning classifiers against state-of-the-art transformer architectures and lexicon-based models.

## Project Overview

- **Dataset**: `AllReviews.xlsx` containing 2000 real amazon product reviews with 1000 being positive and 1000 being negative.
- **Traditional ML Models**:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - Random Forest Classifier
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
- **Deep Learning & Lexicon Approaches**:
  - Pre-trained CardiffNLP Twitter RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`)
  - NLTK VADER (`SentimentIntensityAnalyzer`)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, ROC-AUC curves, and Confusion Matrices.

## Repository Structure

```
SentimentAnalysis/
├── SentimentAnalysis.ipynb   # Main notebook with EDA, ML experiments, and RoBERTa pipeline
├── predict_roberta.py        # Standalone inference script using Hugging Face RoBERTa
├── AllReviews.xlsx           # Customer review dataset
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules for Python, Jupyter checkpoints, and cache
└── README.md                 # Project documentation
```

## Getting Started

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.9+ installed. It is recommended to create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Running the Notebook

Launch Jupyter Notebook or open `SentimentAnalysis.ipynb` in your IDE:

```bash
jupyter notebook SentimentAnalysis.ipynb
```
