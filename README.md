# Medical named entity recognition
This project is intended to test my skills with named entity recognition

The data comes from [pubtator](https://www.ncbi.nlm.nih.gov/research/pubtator/index.html)

## [ODSC approach](./docs/ODSC_approach.md)
Inspiration comes from an [open source course by anujgupta82](https://github.com/anujgupta82/nlp_workshop_odsc_europe21) on NLP:

## [Spacy approach](./docs/spacy_approach.md)
After reading the Spacy documentation I discovered they have some tutorials on their github
https://github.com/explosion/projects

# Setup

## Python dependencies
```bash
pip install -r requirements
```

## Spacy
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

## pre-commit
```bash
pre-commit install
```
