import random
from pathlib import Path
from typing import List

import spacy
import typer
from spacy import Language
from spacy.tokens import DocBin
from tqdm import tqdm

from ner_sentence import NerSentence
from src.definitions import PROJECT_ROOT


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = PROJECT_ROOT / "corpus",
    train_fraction: float = 0.75,
):
    nlp = spacy.blank("en")
    examples = NerSentence.from_pubtator_jsonl(input_path)

    random.shuffle(examples)
    split_index_between_training_and_testing = int(len(examples) * train_fraction)

    train_examples = examples[:split_index_between_training_and_testing]
    output_train = output_path / "train.spacy"
    preprocess_and_persist(nlp, train_examples, output_train)

    eval_examples = examples[split_index_between_training_and_testing:]
    output_eval = output_path / "eval.spacy"
    preprocess_and_persist(nlp, eval_examples, output_eval)


def preprocess_and_persist(
    nlp: Language,
    train_examples: List[NerSentence],
    output_file: Path,
):
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for ner_sentence in tqdm(
        train_examples, desc=f"preprocessing sentences for {output_file}"
    ):
        doc_bin.add(ner_sentence.as_spacy_doc(nlp))
    doc_bin.to_disk(output_file)
    print(f"Processed {len(doc_bin)} documents: {output_file.name}")


if __name__ == "__main__":
    typer.run(main)
