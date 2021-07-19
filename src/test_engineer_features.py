import json
import shutil

import pandas as pd
import pytest
from assertpy import assert_that
from pandas._testing import assert_frame_equal

from src.definitions import PROJECT_ROOT
from src.engineer_features import engineer_features, list_all_pubtator_jsons, extract_annotated_passages, Annotation


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
                {"word": "Formate"},
                {"word": "assay"},
                {"word": "in"},
                {"word": "body"},
                {"word": "fluids"},
                {"word": "application"},
                {"word": "in"},
                {"word": "methanol"},
                {"word": "poisoning"},
            ]
        )
    )


def test_list_all_pubtator_jsons(tmp_path):
    all_pubtator_jsons = list_all_pubtator_jsons(tmp_path)
    assert_that(all_pubtator_jsons).is_length(1)
    assert_that(all_pubtator_jsons[0]).is_type_of(dict)
    assert_that(
        all_pubtator_jsons[0]["passages"]
    ).extracting("text").contains(
        "Formate assay in body fluids: application in methanol poisoning."
    )


def test_extract_annotated_passages():
    annotated_passages = extract_annotated_passages(
        json.loads((PROJECT_ROOT / "data/test/pubtator_example_1.json").open().read())
    )
    assert_that(annotated_passages).is_length(1)
    assert_that(
        annotated_passages[0].text
    ).is_equal_to(
        "Formate assay in body fluids: application in methanol poisoning."
    )
    assert_that(
        annotated_passages[0].annotations
    ).contains(
        Annotation(offset=45, length=8, type="Chemical"),
        Annotation(offset=54, length=9, type="Disease"),
    )
