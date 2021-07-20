import json
from glob import glob
from json import JSONDecodeError
from pathlib import Path
from typing import List

import pandas as pd
import plac
import spacy

from src.definitions import PROJECT_ROOT


def preprocess_data(
    data_root: Path = PROJECT_ROOT / "data/raw",
    output_dir: Path = PROJECT_ROOT / "data/processed",
) -> None:
    examples = read_all_pubtator_examples(data_root)

    nlp = spacy.load("en_core_web_sm")

    df = pd.concat(
        [
            parse_passage(passage, example["id"], idx, nlp)
            for example in examples
            for idx, passage in enumerate(example["passages"])
            if passage["text"] != ""
        ]
    )

    persist(df, output_dir)


def read_all_pubtator_examples(data_root: Path) -> List[dict]:
    examples = [
        parse_pubtator_file(path)
        for path in glob(str(data_root / "**/*.json"), recursive=True)
    ]
    return examples


def parse_pubtator_file(path: Path) -> dict:
    try:
        return json.loads(Path(path).open().read())
    except JSONDecodeError as e:
        raise Exception(f"Failed to parse the pubtator json at: {path=}") from e


def parse_passage(
    passage: dict, pubtator_id: str, passage_id: int, nlp
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pubtator_id": pubtator_id,
                "passage_id": passage_id,
                "words": token.text,
                "labels": [
                    annotation["infons"]["type"]
                    for annotation in passage["annotations"]
                    for location in annotation["locations"]
                    if location["offset"] == token.idx
                ],
            }
            for token in nlp(passage["text"])
        ]
    )


def persist(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    df.to_csv(
        output_dir / "labeled_passages.csv",
        index=False,
    )


if __name__ == "__main__":
    plac.call(preprocess_data)
