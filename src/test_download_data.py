import json
from pathlib import Path

from assertpy import assert_that

from src.download_data import download_single_abstract, save_to_corpus


def test_download_single_abstract():
    parsed = json.loads(download_single_abstract(1))
    assert_that(parsed).has_id("1")
    assert_that(parsed["passages"]).extracting("text").contains(
        "Formate assay in body fluids: application in methanol poisoning."
    )


def test_save_to_corpus(tmp_path: Path):
    content = '{"content": "Fake example for test_save_to_corpus"}'
    save_to_corpus(
        abstract=content,
        abstract_id=1,
        project_root=tmp_path,
    )
    with (tmp_path / "data/raw/0to99/1.json").open() as file:
        assert_that(file.read()).is_equal_to(content)
