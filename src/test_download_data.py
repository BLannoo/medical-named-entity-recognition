import json

from assertpy import assert_that

from src.download_data import download_single_abstract


def test_download_single_abstract():
    example = json.loads(download_single_abstract(1))
    assert_that(example).has_id("1")
    assert_that(
        example["passages"]
    ).extracting(
        "text"
    ).contains(
        "Formate assay in body fluids: application in methanol poisoning."
    )
