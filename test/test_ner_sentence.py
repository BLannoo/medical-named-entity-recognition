import assertpy

from ner_sentence import NerSentence, NerSpan
from src.definitions import PROJECT_ROOT


def test_from_pubtator_jsonl():
    assertpy.assert_that(
        NerSentence.from_pubtator_jsonl(PROJECT_ROOT / "data/test/raw/dummy.jsonl")
    ).is_length(3).does_not_contain(NerSentence(text="", entities=tuple())).contains(
        NerSentence(
            text="A dummy title, with a dummyentity",
            entities=(NerSpan(start_char=22, end_char=33, label="dummy-label"),),
        )
    ).contains(
        NerSentence(
            text="A dummy abstract, with a dummyentity",
            entities=(NerSpan(start_char=25, end_char=36, label="dummy-label"),),
        )
    ).contains(
        NerSentence(text="Another dummy title", entities=tuple()),
    )
