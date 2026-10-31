# NLP-for-Healthcare

Rule-based **medical entity tagging** for healthcare text: a simple baseline for clinical NLP using a term dictionary and regular expressions.

> Status: baseline example, not validated. Not for clinical use.

## What it does
`NamedEntityRecognition(NER).py` loads the CSV tables of the MIMIC-III database and tags entities in the `long_title` column of `D_ICD_PROCEDURES.csv` (ICD procedure descriptions):
- **Dictionary matching:** a small list of terms for three types, *Drug*, *Condition* and *Procedure* (case-insensitive).
- **Regular expression:** dates in `YYYY-MM-DD` format.

The script prints each title with its extracted entities.

## Limitations
- The dictionary is small, and matching is plain substring search, so it can miss terms or match inside longer words.
- It is not a machine-learning NER model, and it has not been evaluated (no precision, recall or F1).

## Data
MIMIC-III is **not included**. It requires credentialed access through PhysioNet and a data use agreement. Never upload MIMIC data, or any patient data, to a public repository.

## Requirements
`pip install pandas`. Set `data_path` at the top of the script to your local MIMIC-III folder.

## Planned
Use a trained biomedical NER model (for example spaCy-based) and evaluate it with precision, recall and F1 on annotated text.

## Use and contact
If you use, adapt or contribute to this work, please credit the author and **contact or inform me by email: owaiskhan_bangash@yahoo.com**

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
