from pathlib import Path

import pandas as pd
from assertpy import assert_that

from src.definitions import PROJECT_ROOT
from src.preprocess_data import preprocess_data


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
            "labels": "B-Chemical",
            "passage_id": 0,
            "pubtator_id": 1,
            "words": "methanol",
        }
    )
    assert_that(word_as_dict("poisoning")).is_equal_to(
        {
            "labels": "B-Disease",
            "passage_id": 0,
            "pubtator_id": 1,
            "words": "poisoning",
        }
    )
    assert_that(word_as_dict("pyridine")).is_equal_to(
        {
            "labels": "B-Chemical",
            "passage_id": 0,
            "pubtator_id": 2,
            "words": "pyridine",
        }
    )
    assert_that(word_as_dict("nucleotide")).is_equal_to(
        {
            "labels": "I-Chemical",
            "passage_id": 0,
            "pubtator_id": 2,
            "words": "nucleotide",
        }
    )

    expected = pd.read_csv(PROJECT_ROOT / "data/test/expected_labeled_passages.csv")
    pd.testing.assert_frame_equal(left=actual, right=expected)
