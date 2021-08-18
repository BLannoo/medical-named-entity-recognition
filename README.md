# Medical named entity recognition

This project is intended to test my skills with named entity recognition

Inspiration comes from combining a open source course by anujgupta82 on NLP
and the open source data set from pubtator:
* https://github.com/anujgupta82/nlp_workshop_odsc_europe21
* https://www.ncbi.nlm.nih.gov/research/pubtator/index.html

# Setup

## Spacy
```bash
pip install -r requirements
python -m spacy download en_core_web_sm
```

## pre-commit
```bash
pip install -r requirements
pre-commit install
```

# Execution

Download 100 batches of 1000 examples from pubtator
```bash
python src/main/download_data.py 100
```

Preprocess the data into a csv for each word from each fragment
```bash
python src/main/preprocess_data.py
```
