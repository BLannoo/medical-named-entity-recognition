from pathlib import Path

import assertpy
import spacy
from spacy.tokens import DocBin

from preprocess import main
from src.definitions import PROJECT_ROOT


def test_preprocess(tmp_path: Path):
    main(
        input_path=PROJECT_ROOT / "data/test/raw/dummy.jsonl",
        output_path=tmp_path,
        train_fraction=0.5,
    )

    nlp = spacy.blank("en")

    train_docs = DocBin().from_disk(tmp_path / "train.spacy").get_docs(nlp.vocab)
    eval_docs = DocBin().from_disk(tmp_path / "eval.spacy").get_docs(nlp.vocab)

    assertpy.assert_that([doc.text for doc in [*train_docs, *eval_docs]]).is_length(
        3
    ).contains("A dummy title, with a dummyentity").contains(
        "A dummy abstract, with a dummyentity"
    ).contains(
        "Another dummy title"
    )
