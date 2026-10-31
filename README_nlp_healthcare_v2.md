# NLP-for-Healthcare

Example scripts for **healthcare text mining**: text classification, topic modeling, named entity recognition (NER) and sentiment analysis, using classical machine learning and deep learning.

> Status: baseline example scripts, not validated. **Not for clinical use.**

## Contents

| Folder | What it contains |
|---|---|
| `02- Text Classification Algorithms/` | BERT-based text classification (`01- BERT.py`, `02 Bert-mod.py`, `PreTrain.py`) |
| `04- Topic Modeling/` | LDA, LSA, NMF and HDP topic models |
| `Named Entity Recognition (NER)/` | Rule-based tagging on MIMIC-III tables (`01.py`, `02.py`, `03-01.py`) and a CRF demo (`03.py`) |
| `06- Sentiment analysis and emotion detection/` | Traditional ML (Naive Bayes, SVM, TF-IDF, SMOTE), deep learning (RNN, LSTM, BiLSTM, BiLSTM with attention), and a lexicon-based method |

## Data (not included)
- **MIMIC-III** needs credentialed access through PhysioNet and a data use agreement. Never upload MIMIC data, or any patient data, to a public repository.
- The classification and sentiment scripts read a Chinese-language CSV of clinical-department text (`外科5-14000.csv`). It is not included. Use your own CSV with a text column and a label column, and set its path at the top of each script.
- BERT scripts need a local copy of `bert-base-uncased` (download it from Hugging Face). The weights are not included.

## Requirements
Python 3.9+ and, depending on the script: `pandas numpy scikit-learn imbalanced-learn gensim tensorflow transformers torch sklearn-crfsuite`.

## Run
1. Set the data path at the top of the script (the paths in the code are examples from the author's computer).
2. Run one script at a time, for example `python "04- Topic Modeling/LDA.py"`.

## Limitations
- No results are reported in this repository, and the scripts have not been systematically evaluated.
- The CRF NER script (`03.py`) trains on two hand-written example sentences. It is a demo of the pipeline only.
- The rule-based NER uses a small term list and plain substring matching.

## Planned
Trained biomedical NER models (for example spaCy-based), cleaned command-line arguments, and evaluation with precision, recall and F1.

## Use and contact
If you use, adapt or contribute to this work, please credit the author and **contact or inform me by email: owaiskhan_bangash@yahoo.com**

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
