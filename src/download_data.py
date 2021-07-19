from pathlib import Path

import plac
import requests
from tqdm import tqdm

from src.definitions import PROJECT_ROOT


def download_data(
        num_examples: int = 100,
):
    for abstract_id in tqdm(range(1, num_examples)):
        abstract = download_single_abstract(abstract_id)
        save_to_corpus(abstract, abstract_id)


def save_to_corpus(
        abstract: str,
        abstract_id: int,
        project_root: Path = PROJECT_ROOT,
) -> None:
    dir_size = 100
    dir_start = abstract_id // dir_size * dir_size
    dir = Path(project_root / f"data/raw/{dir_start}to{dir_start + dir_size - 1}")
    dir.parent.parent.mkdir(exist_ok=True)
    dir.parent.mkdir(exist_ok=True)
    dir.mkdir(exist_ok=True)
    file_path = dir / f"{abstract_id}.json"
    file_path.open("w").write(abstract)


def download_single_abstract(abstract_id: int) -> str:
    response = requests.post(
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson",
        json={"pmids": [abstract_id]},
    )
    if response.status_code != 200:
        print(f"[Error]: HTTP code {response.status_code} for {abstract_id=}")
    elif response.text == "":
        print(f"[Error]: empty body for {abstract_id=}")
    else:
        return response.text


if __name__ == "__main__":
    plac.call(download_data)
