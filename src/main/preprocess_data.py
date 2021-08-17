import json
from json import JSONDecodeError
from pathlib import Path
from typing import List

import pandas as pd
import plac
import spacy
from tqdm import tqdm

from src.definitions import PROJECT_ROOT


def main(
    data_root: Path = PROJECT_ROOT / "data/raw/pubtator",
    output_dir: Path = PROJECT_ROOT / "data/processed",
) -> None:
    preprocess_data(data_root, output_dir)


def preprocess_data(
    data_root: Path,
    output_dir: Path,
) -> None:
    examples = read_all_pubtator_examples(data_root)

    nlp = spacy.load("en_core_web_sm")

    df = pd.concat(
        [
            parse_passage(passage, example["id"], idx, nlp)
            for example in tqdm(examples)
            for idx, passage in enumerate(example["passages"])
            if passage["text"] != ""
        ]
    )

    persist(df, output_dir)


def read_all_pubtator_examples(data_root: Path) -> List[dict]:
    examples = [
        parse_pubtator_file(line)
        for path in data_root.glob("**/*.jsonl")
        for line in path.read_text().strip().split("\n")
    ]
    return examples


def parse_pubtator_file(json_line: str) -> dict:
    try:
        return json.loads(json_line)
    except JSONDecodeError as e:
        raise Exception(f"Failed to parse the pubtator json: {json_line=}") from e


def parse_passage(
    passage: dict, pubtator_id: str, passage_id: int, nlp
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pubtator_id": pubtator_id,
                "passage_id": passage_id,
                "words": token.text,
                "POS": token.pos_,
                "labels": determine_label(token, passage),
            }
            for token in nlp(passage["text"])
        ]
    )


def determine_label(token, passage):
    all_labels = [
        ("B-" if location["offset"] == token.idx else "I-")
        + annotation["infons"]["type"]
        for annotation in passage["annotations"]
        for location in annotation["locations"]
        if (location["offset"] <= token.idx < location["offset"] + location["length"])
    ]
    if len(all_labels) == 0:
        return "O"
    if len(all_labels) == 1:
        return all_labels[0]
    raise Exception(
        f"'{token.text}' matched with multiple "
        f"labels '{all_labels}' in passage '{passage}'"
    )


def persist(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    df.to_csv(
        output_dir / "labeled_passages.csv",
        index=False,
    )


if __name__ == "__main__":
    plac.call(main)
