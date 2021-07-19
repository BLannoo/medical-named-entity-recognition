import json
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plac
import spacy
from spacy.tokens.token import Token

from src.definitions import PROJECT_ROOT
from src.logger import logger


def preprocess_data(project_root: Path = PROJECT_ROOT):
    df_tokens_with_features = pd.DataFrame(
        asdict(engineer_features(token_with_context))
        for pubtator_json in list_all_pubtator_jsons(project_root)
        for annotated_passage in extract_annotated_passages(pubtator_json)
        for token_with_context in generate_all_annotated_tokens(annotated_passage)
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
                logger.warn(
                    f"multiple locations not supported for now, example: {pubtator_json}"
                )
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
            ],
        )
        for passage in pubtator_json["passages"]
        if passage["text"] != ""
    ]


@dataclass(frozen=True)
class AnnotatedToken:
    token: Token
    ner_type: Optional[str]

    @staticmethod
    def create(token: Token, annotated_passage: AnnotatedPassage) -> "AnnotatedToken":
        ner_types = [
            annotation.type
            for annotation in annotated_passage.annotations
            if annotation.offset == token.idx
        ]
        return AnnotatedToken(
            token=token,
            ner_type=ner_types[0] if ner_types else None,
        )


def generate_all_annotated_tokens(
    annotated_passage: AnnotatedPassage,
) -> List[AnnotatedToken]:
    nlp = spacy.load("en_core_web_sm")
    return [
        AnnotatedToken.create(token, annotated_passage)
        for token in nlp(annotated_passage.text)
    ]


@dataclass(frozen=True)
class TokenWithFeatures:
    token: str


def engineer_features(annotated_passage: AnnotatedToken) -> TokenWithFeatures:
    pass


def persist_tokens_with_features(df: pd.DataFrame, project_root: Path) -> None:
    (project_root / "data").mkdir(exist_ok=True)
    (project_root / "data/processed").mkdir(exist_ok=True)
    df.to_csv(project_root / "data/processed/token_with_features.csv", index=False)


if __name__ == "__main__":
    plac.call(engineer_features)
