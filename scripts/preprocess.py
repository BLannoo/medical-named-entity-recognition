import random
from pathlib import Path
from typing import List

import spacy
import srsly
import typer
from spacy.tokens import DocBin, Span, Doc

from src.definitions import PROJECT_ROOT


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = PROJECT_ROOT / "corpus",
    train_fraction: float = 0.75,
):
    nlp = spacy.blank("en")
    examples = list(srsly.read_jsonl(input_path))

    random.shuffle(examples)
    split_index_between_training_and_testing = int(len(examples) * train_fraction)

    train_examples = examples[:split_index_between_training_and_testing]
    output_train = output_path / "train.spacy"
    preprocess_and_persist(nlp, train_examples, output_train)

    eval_examples = examples[split_index_between_training_and_testing:]
    output_eval = output_path / "eval.spacy"
    preprocess_and_persist(nlp, eval_examples, output_eval)


def preprocess_and_persist(nlp, train_examples, output_file):
    doc_bin_train = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for example in train_examples:
        for doc in process_passages(example, nlp):
            doc_bin_train.add(doc)
    doc_bin_train.to_disk(output_file)
    print(f"Processed {len(doc_bin_train)} documents: {output_file.name}")


def process_passages(example, nlp) -> List[spacy.tokens.doc.Doc]:
    docs = []
    for passage in example["passages"]:
        if passage["text"] != "":
            doc = nlp(passage["text"])

            entities = []
            for annotation in passage["annotations"]:
                span = annotation_to_span(doc, annotation, passage["offset"])
                if span is not None:
                    entities.append(span)

            doc.ents = entities

            docs.append(doc)
    return docs


def annotation_to_span(doc: Doc, annotation: dict, passage_offset: int) -> Span:
    location = annotation["locations"][0]
    start = location["offset"] - passage_offset
    return doc.char_span(
        start,
        start + location["length"],
        label=annotation["infons"]["type"],
    )


if __name__ == "__main__":
    typer.run(main)
