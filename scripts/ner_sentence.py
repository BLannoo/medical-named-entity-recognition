from pathlib import Path
from typing import Tuple, List, Optional

import pydantic
from spacy import Language
from spacy.tokens import Doc, Span

from pubtator_pydantic import Pubtator, Annotation


class NerSpan(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    start_char: int
    end_char: int
    label: str

    @staticmethod
    def from_pubtator_annotation(
        annotation: Annotation, passage_offset: int
    ) -> "NerSpan":
        start_char = annotation.locations[0].offset - passage_offset
        return NerSpan(
            start_char=start_char,
            end_char=start_char + annotation.locations[0].length,
            label=annotation.infons.type,
        )

    def as_spacy_span(self, doc: Doc) -> Optional[Span]:
        return doc.char_span(
            self.start_char,
            self.end_char,
            label=self.label,
        )


class NerSentence(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    text: str
    entities: Tuple[NerSpan, ...]

    @staticmethod
    def from_pubtator_jsonl(file_path: Path) -> List["NerSentence"]:
        return [
            NerSentence(
                text=passage.text,
                entities=tuple(
                    NerSpan.from_pubtator_annotation(annotation, passage.offset)
                    for annotation in passage.annotations
                ),
            )
            for example in file_path.read_text().strip().split("\n")
            for passage in Pubtator.parse_raw(example).passages
            if passage.text != ""
        ]

    def as_spacy_doc(self, nlp: Language) -> Doc:
        doc = nlp(self.text)
        spans = []
        for entity in self.entities:
            span = entity.as_spacy_span(doc)
            if span is not None:
                spans.append(span)
        doc.ents = spans
        return doc
