import json
from pathlib import Path

import spacy
from spacy import displacy

from preprocess import process_passages


def test_1():
    nlp = spacy.blank("en")
    example = json.loads(Path("example_999.json").read_text())
    docs = process_passages(example, nlp)
    displacy.serve(docs, style="ent")
