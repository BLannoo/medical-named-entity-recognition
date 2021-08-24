from pathlib import Path

import pandas as pd
import spacy
from assertpy import assert_that

from src.definitions import PROJECT_ROOT
from src.main.preprocess_data import preprocess_data, parse_passage


def test_preprocess_data(tmp_path: Path):
    preprocess_data(
        data_root=PROJECT_ROOT / "data/test/raw",
        output_dir=tmp_path,
    )

    actual = pd.read_csv(tmp_path / "labeled_passages.csv")

    def word_as_dict(word: str) -> dict:
        return actual[actual.words == word].iloc[0].to_dict()

    assert_that(word_as_dict("methanol")).is_equal_to(
        {
            "passage_id": 0,
            "pubtator_id": 1,
            "words": "methanol",
            "POS": "NOUN",
            "labels": "B-Chemical",
        }
    )
    assert_that(word_as_dict("poisoning")).is_equal_to(
        {
            "passage_id": 0,
            "pubtator_id": 1,
            "words": "poisoning",
            "POS": "NOUN",
            "labels": "B-Disease",
        }
    )
    assert_that(word_as_dict("pyridine")).is_equal_to(
        {
            "passage_id": 0,
            "pubtator_id": 2,
            "words": "pyridine",
            "POS": "NOUN",
            "labels": "B-Chemical",
        }
    )
    assert_that(word_as_dict("nucleotide")).is_equal_to(
        {
            "passage_id": 0,
            "pubtator_id": 2,
            "words": "nucleotide",
            "POS": "NOUN",
            "labels": "I-Chemical",
        }
    )

    expected = pd.read_csv(PROJECT_ROOT / "data/test/labeled_passages.csv")
    pd.testing.assert_frame_equal(left=actual, right=expected)


def test_parse_passage_can_handle_global_offset():
    nlp = spacy.load("en_core_web_sm")

    actual = parse_passage(
        passage={
            "offset": 20,  # This is the parameter under test
            "text": "Adsorption of rRNA and poly(A)-containing RNA to filters.",
            "annotations": [
                {
                    "infons": {"identifier": "MESH:D011061", "type": "Chemical"},
                    # TODO: configure tokenization to split on '-'
                    "text": "poly(A)",
                    "locations": [{"offset": 43, "length": 7}],
                },
            ],
        },
        pubtator_id="0",
        passage_id=0,
        nlp=nlp,
    )

    print(actual)

    expected = pd.DataFrame(
        {
            "pubtator_id": ["0"] * 9,
            "passage_id": [0] * 9,
            "words": [
                "Adsorption",
                "of",
                "rRNA",
                "and",
                "poly(A)-containing",
                "RNA",
                "to",
                "filters",
                ".",
            ],
            "POS": [
                "NOUN",
                "ADP",
                "ADJ",
                "CCONJ",
                "VERB",
                "PROPN",
                "ADP",
                "NOUN",
                "PUNCT",
            ],
            "labels": ["O"] * 4 + ["B-Chemical"] + ["O"] * 4,
        }
    )

    pd.testing.assert_frame_equal(left=actual, right=expected)
