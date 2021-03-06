"""Visualize the data with Streamlit and spaCy."""
from pathlib import Path
from typing import Tuple, List, Set

import spacy
import streamlit as st
import typer
from spacy import displacy

from ner_sentence import NerSentence


@st.cache(allow_output_mutation=True)
def load_data(filepath: Path) -> Tuple[List[dict], Set[str], int, int]:
    sentences = NerSentence.from_pubtator_jsonl(filepath)
    rows = []
    n_total_ents = 0
    n_no_ents = 0
    labels = set()
    nlp = spacy.blank("en")
    for sentence in sentences:
        doc = sentence.as_spacy_doc(nlp)
        ents = [
            {
                "start": span.start_char,
                "end": span.end_char,
                "token_start": span.start,
                "token_end": span.end,
                "label": span.label_,
            }
            for span in doc.ents
        ]
        row = {"text": doc.text, "ents": ents}
        n_total_ents += len(row["ents"])
        if not row["ents"]:
            n_no_ents += 1
        labels.update([span["label"] for span in row["ents"]])
        rows.append(row)
    return rows, labels, n_total_ents, n_no_ents


def main(file_paths: str):
    files = [p.strip() for p in file_paths.split(",")]
    st.sidebar.title("Data visualizer")
    st.sidebar.markdown(
        "Visualize the annotations using "
        "[displaCy](https://spacy.io/usage/visualizers) "
        "and view stats about the datasets."
    )
    data_file = st.sidebar.selectbox("Dataset", files)
    data, labels, n_total_ents, n_no_ents = load_data(Path(data_file))
    displacy_settings = {
        "style": "ent",
        "manual": True,
        "options": {"colors": {label: "#d1bcff" for label in labels}},
    }
    st.header(f"{data_file} ({len(data)})")
    wrapper = "<div style='border-bottom: 1px solid #ccc; padding: 20px 0'>{}</div>"
    MAX_EXAMPLES_DISPLAYED = 100
    for row in data[:MAX_EXAMPLES_DISPLAYED]:
        html = displacy.render(row, **displacy_settings).replace("\n\n", "\n")
        st.markdown(wrapper.format(html), unsafe_allow_html=True)
    if len(data) > MAX_EXAMPLES_DISPLAYED:
        st.markdown(
            f"Only displays the first {MAX_EXAMPLES_DISPLAYED} examples"
            f" out of {len(data)}"
        )

    st.sidebar.markdown(
        f"""
    | `...{data_file[-20:]}` | |
    | --- | ---: |
    | Total examples | {len(data):,} |
    | Total entities | {n_total_ents:,} |
    | Examples with no entities | {n_no_ents:,} |
    """
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
