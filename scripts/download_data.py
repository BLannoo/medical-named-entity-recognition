from pathlib import Path
from typing import List, Tuple

import requests
import typer
from tqdm import tqdm

from src.definitions import PROJECT_ROOT


def main(
    last_id: int = 1_000, destination_folder: Path = PROJECT_ROOT / "data/raw/pubtator"
):
    example_batches = [
        download_batch(batch_start_id, batch_end_id)
        for batch_start_id, batch_end_id in tqdm(
            generate_batches(last_id), desc="downloading 1_000 examples per iteration"
        )
    ]
    destination_folder.mkdir(exist_ok=True)
    destination_file = destination_folder / f"from1to{last_id}.jsonl"
    destination_file.write_text("\n".join(example_batches))


def generate_batches(last_id: int) -> List[Tuple[int, int]]:
    return [
        (first_id_batch, min(first_id_batch + 999, last_id))
        for first_id_batch in range(1, last_id, 1000)
    ]


def download_batch(batch_start_id: int, batch_end_id: int) -> str:
    try:
        response = requests.post(
            "https://www.ncbi.nlm.nih.gov/"
            "research/pubtator-api/publications/export/biocjson",
            json={"pmids": list(range(batch_start_id, batch_end_id + 1))},
        )
        if response.status_code != 200:
            print(
                f"[Error]: {response.status_code=} "
                f"for {batch_start_id=} and {batch_end_id=} "
                f"with {response.text=}"
            )
        elif response.text == "":
            print(f"[Error]: empty body for {batch_start_id=} and {batch_end_id=}")
        else:
            return response.text.strip()
    except ConnectionError as e:
        print(f"[Exception]: for {batch_start_id=} and {batch_end_id=}: {e}")


if __name__ == "__main__":
    typer.run(main)
