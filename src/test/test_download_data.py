import json
from pathlib import Path

import pytest
from assertpy import assert_that

from src.main.download_data import download_single_batch, download_from_to


def test_download_first_batch(tmp_path: Path):
    # When
    download_single_batch(
        location=tmp_path,
        start_id=0,
    )

    # Then
    location = tmp_path / "0to999.jsonl"
    assert location.exists()

    downloaded_dicts = [
        json.loads(json_line) for json_line in location.read_text().strip().split("\n")
    ]

    assert_that(downloaded_dicts).is_length(999)
    assert_that(downloaded_dicts).extracting("id").contains("1", "999")
    assert_that(
        [
            passage["text"]
            for downloaded_dict in downloaded_dicts
            for passage in downloaded_dict["passages"]
        ]
    ).contains("Formate assay in body fluids: application in methanol poisoning.")


def test_non_existing_location():
    with pytest.raises(
        ValueError,
        match="/non/existing/path",
    ):
        download_single_batch(
            location=Path("/non/existing/path"),
            start_id=1,
        )


def test_download_from_to(tmp_path: Path):
    # When
    download_from_to(
        location=tmp_path,
        first_batch_id=0,
        number_of_batches=2,
    )

    # Then
    assert (tmp_path / "0to999.jsonl").exists()
    assert (tmp_path / "1000to1999.jsonl").exists()
