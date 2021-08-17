from pathlib import Path

import plac
import requests
from tqdm import tqdm

from src.definitions import PROJECT_ROOT
from src.logger import logger

PUBTATOR_MAX_BATCH_SIZE = 1000


def main(
    number_of_batches: int = 10,
    first_batch_id: int = 0,
    location: Path = PROJECT_ROOT / "data/raw/pubtator",
) -> None:
    download_from_to(number_of_batches, first_batch_id, location)


def download_from_to(
    number_of_batches: int,
    first_batch_id: int,
    location: Path,
) -> None:
    location.mkdir(exist_ok=True)
    for batch_id in tqdm(range(first_batch_id, first_batch_id + number_of_batches)):
        download_single_batch(location, batch_id * PUBTATOR_MAX_BATCH_SIZE)


def download_single_batch(
    location: Path,
    start_id: int,
) -> None:
    if not location.exists():
        raise ValueError(f"'{location=}' must exist already")

    folder_size = PUBTATOR_MAX_BATCH_SIZE * 10
    folder_start_id = start_id // folder_size * folder_size
    folder_end_id = folder_start_id + folder_size - 1
    folder = location / f"{folder_start_id}to{folder_end_id}"
    folder.mkdir(exist_ok=True)

    end_id = start_id + PUBTATOR_MAX_BATCH_SIZE - 1
    file_location = folder / f"{start_id}to{end_id}.jsonl"

    try:
        response = requests.post(
            "https://www.ncbi.nlm.nih.gov/"
            "research/pubtator-api/publications/export/biocjson",
            json={"pmids": list(range(start_id, end_id + 1))},
        )

        if response.status_code != 200:
            print(
                f"[Error]: {response.status_code=} for {start_id=},"
                f" with {response.text=}"
            )
        elif response.text == "":
            print(f"[Error]: empty body for {start_id=}")
        else:
            file_location.write_text(response.text)
            logger.info(f"Downloaded data to {file_location=}")

    except ConnectionError as e:
        print(f"[Exception]: for {start_id=}: {e}")


if __name__ == "__main__":
    plac.call(main)
