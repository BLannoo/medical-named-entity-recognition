import json
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
import plac

from src.definitions import PROJECT_ROOT
from src.logger import logger


def engineer_features(
        project_root: Path = PROJECT_ROOT
):
    df_tokens_with_features = pd.DataFrame(
        asdict(token_with_features)
        for pubtator_json in list_all_pubtator_jsons(project_root)
        for annotated_passage in extract_annotated_passages(pubtator_json)
        for token_with_features in feature_engineered(annotated_passage)
    )
    persist_tokens_with_features(df_tokens_with_features, project_root)


def list_all_pubtator_jsons(project_root: Path) -> List[dict]:
    return [
        json.loads(Path(path).open().read())
        for path in glob(str(project_root / "data/raw/*/*.json"))
    ]


@dataclass(frozen=True)
class Annotation:
    offset: int
    length: int
    type: str


@dataclass(frozen=True)
class AnnotatedPassage:
    text: str
    annotations: List[Annotation]


def extract_annotated_passages(pubtator_json: dict) -> List[AnnotatedPassage]:

    # TODO: handle multiple locations for a single annotation
    for passage in pubtator_json["passages"]:
        for annotation in passage["annotations"]:
            if len(annotation["locations"]) != 1:
                logger.warn(f"multiple locations not supported for now, example: {pubtator_json}")
                return []

    return [
        AnnotatedPassage(
            text=passage["text"],
            annotations=[
                Annotation(
                    offset=annotation["locations"][0]["offset"],
                    length=annotation["locations"][0]["length"],
                    type=annotation["infons"]["type"],
                )
                for annotation in passage["annotations"]
            ]
        )
        for passage in pubtator_json["passages"]
        if passage["text"] != ""
    ]


@dataclass(frozen=True)
class TokenWithFeatures:
    pass


def feature_engineered(annotated_passage: AnnotatedPassage) -> List[TokenWithFeatures]:
    pass


def persist_tokens_with_features(df: pd.DataFrame, project_root: Path) -> None:
    (project_root / "data").mkdir(exist_ok=True)
    (project_root / "data/processed").mkdir(exist_ok=True)
    df.to_csv(project_root / "data/processed/token_with_features.csv", index=False)


if __name__ == '__main__':
    plac.call(engineer_features)
