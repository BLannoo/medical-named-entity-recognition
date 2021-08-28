from pathlib import Path
from typing import Tuple, List

import spacy
import srsly
import typer
from spacy.tokens import DocBin, Span, Doc


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
):
    nlp = spacy.blank("en")
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    skipped_annotations = 0
    for example in srsly.read_jsonl(input_path):
        docs, skipped_annotations_single_example = process_passages(example, nlp)
        skipped_annotations += skipped_annotations_single_example
        for doc in docs:
            doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Processed {len(doc_bin)} documents: {output_path.name}")
    print(f"{skipped_annotations=}")


def process_passages(example, nlp) -> Tuple[List[spacy.tokens.doc.Doc], int]:
    docs = []
    skipped_annotations = 0
    for passage in example["passages"]:
        if passage["text"] != "":
            doc = nlp(passage["text"])

            entities = []
            for annotation in passage["annotations"]:
                span = annotation_to_span(doc, annotation, passage["offset"])
                if span is not None:
                    entities.append(span)
                else:
                    skipped_annotations += 1

            doc.ents = entities

            docs.append(doc)
    return docs, skipped_annotations


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
