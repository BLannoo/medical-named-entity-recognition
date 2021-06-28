import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable
import matplotlib.pyplot as plt

import plac

from logger import logger


def main():
    corpus = [
        text
        for folder in os.listdir("data")
        for filename in os.listdir(Path("data") / folder)
        for text in extract_data(Path("data") / folder / filename)
    ]

    plt.hist(
        [len(case.text) for case in corpus],
        bins=100,
    )
    plt.show()


@dataclass(frozen=True)
class Annotation:
    offset: int
    length: int
    type: str
    text: str

    @staticmethod
    def validate(annotation_as_dict: Dict[str, str]) -> bool:
        validations = (
            lambda content: "locations" in annotation_as_dict,
            lambda content: len(annotation_as_dict["locations"]) == 1,
            lambda content: "offset" in annotation_as_dict["locations"][0],
            lambda content: "length" in annotation_as_dict["locations"][0],
            lambda content: "infons" in annotation_as_dict,
            lambda content: "type" in annotation_as_dict["infons"],
            lambda content: "text" in annotation_as_dict,
        )
        return check(annotation_as_dict, validations)

    @staticmethod
    def validate_and_create(annotation_as_dict: dict) -> "Annotation":
        return Annotation(
            offset=int(annotation_as_dict["locations"][0]["offset"]),
            length=int(annotation_as_dict["locations"][0]["length"]),
            type=annotation_as_dict["infons"]["type"],
            text=annotation_as_dict["text"],
        )


@dataclass(frozen=True)
class Text:
    text: str
    annotations: Tuple[Annotation]

    @staticmethod
    def validate(passage_as_dict: Dict[str, str]) -> bool:
        validations = (
            lambda content: "text" in passage_as_dict,
            lambda content: "annotations" in passage_as_dict,
        )
        return check(passage_as_dict, validations)

    @staticmethod
    def validate_and_create(passage_as_dict: dict) -> "Text":
        Text.validate(passage_as_dict)
        text = Text(
            text=passage_as_dict["text"],
            annotations=tuple(
                Annotation.validate_and_create(annotation)
                for annotation in passage_as_dict["annotations"]
            ),
        )
        logger.info(f"created Text: {text}")
        return text


def check(as_dict: dict, validations: Tuple):
    is_valid = True
    for validation in validations:
        if not validation(as_dict):
            is_valid = False
            source = inspect.getsource(validation).strip().strip(",")
            logger.warning(f"Validation failed '{source}' content was: {as_dict}")
    return is_valid


def extract_data(path: Path) -> Iterable[Text]:
    with open(path) as file:
        return (
            Text.validate_and_create(passage) for passage in json.load(file)["passages"]
        )


if __name__ == "__main__":
    plac.call(main)
