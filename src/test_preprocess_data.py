import json
import shutil
from typing import List, Optional

import pandas as pd
import pytest
from assertpy import assert_that
from pandas._testing import assert_frame_equal

from src.definitions import PROJECT_ROOT
from src.preprocess_data import (
    engineer_features,
    list_all_pubtator_jsons,
    extract_annotated_passages,
    Annotation,
    AnnotatedPassage,
    generate_all_annotated_tokens,
)


@pytest.fixture(autouse=True)
def setup_raw_data_on_tmp_path(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/raw").mkdir()
    (tmp_path / "data/raw/0to99").mkdir()
    shutil.copy(
        PROJECT_ROOT / "data/test/pubtator_example_1.json",
        tmp_path / "data/raw/0to99/1.json",
    )


def test_engineer_features(tmp_path):
    engineer_features(project_root=tmp_path)

    assert_frame_equal(
        left=pd.read_csv(PROJECT_ROOT / "data/processed/token_with_features.csv"),
        right=pd.DataFrame(
            [
                {"token": "Formate"},
                {"token": "assay"},
                {"token": "in"},
                {"token": "body"},
                {"token": "fluids"},
                {"token": "application"},
                {"token": "in"},
                {"token": "methanol"},
                {"token": "poisoning"},
            ]
        ),
    )


def test_list_all_pubtator_jsons(tmp_path):
    all_pubtator_jsons = list_all_pubtator_jsons(tmp_path)
    assert_that(all_pubtator_jsons).is_length(1)
    assert_that(all_pubtator_jsons[0]).is_type_of(dict)
    assert_that(all_pubtator_jsons[0]["passages"]).extracting("text").contains(
        "Formate assay in body fluids: application in methanol poisoning."
    )


EXAMPLE_ANNOTATED_PASSAGE = AnnotatedPassage(
    text="Formate assay in body fluids: application in methanol poisoning.",
    annotations=[
        Annotation(offset=45, length=8, type="Chemical"),
        Annotation(offset=54, length=9, type="Disease"),
    ],
)


def test_extract_annotated_passages():
    annotated_passages = extract_annotated_passages(
        json.loads((PROJECT_ROOT / "data/test/pubtator_example_1.json").open().read())
    )
    assert_that(annotated_passages).is_length(1)
    assert_that(annotated_passages[0]).is_equal_to(EXAMPLE_ANNOTATED_PASSAGE)


def test_generate_all_tokens_with_context():
    all_annotated_tokens = generate_all_annotated_tokens(EXAMPLE_ANNOTATED_PASSAGE)

    def check_tokens_by_ner_type(ner_type: Optional[str], tokens: List[str]) -> None:
        assert_that(all_annotated_tokens).extracting(
            "token", filter=lambda x: x.ner_type is ner_type
        ).extracting("text").contains(*tokens)

    check_tokens_by_ner_type(
        ner_type=None,
        tokens=["Formate", "assay", "in", "body", "fluids", "application", "in"],
    )

    check_tokens_by_ner_type(ner_type="Chemical", tokens=["methanol"])

    check_tokens_by_ner_type(ner_type="Disease", tokens=["poisoning"])
