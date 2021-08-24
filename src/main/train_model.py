from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import plac
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics as crf_metrics

from src.definitions import PROJECT_ROOT


def main(
    labeled_passages_file_name: Path = PROJECT_ROOT
    / "data/processed/labeled_passages.csv",
    model_output_file_name: Path = PROJECT_ROOT / "output/ner_model.pkl",
    report_output_file_name: Path = PROJECT_ROOT / "output/report.txt",
):
    train_model(
        labeled_passages_file_name=labeled_passages_file_name,
        model_output_file_name=model_output_file_name,
        report_output_file_name=report_output_file_name,
    )


def train_model(
    labeled_passages_file_name: Path,
    model_output_file_name: Path,
    report_output_file_name: Path,
):
    X, y = engineer_features(labeled_passages_file_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True,
    )

    crf.fit(X_train, y_train)

    joblib.dump(crf, model_output_file_name)

    evaluate(crf, X_test, y_test, report_output_file_name)


def engineer_features(labeled_passages_file_name):
    grouped_df = (
        pd.read_csv(labeled_passages_file_name)
        .groupby(["pubtator_id", "passage_id"])
        .apply(
            lambda s: [
                (w, p, t)
                for w, p, t in zip(
                    s["words"].values.tolist(),
                    s["POS"].values.tolist(),
                    s["labels"].values.tolist(),
                )
            ]
        )
    )
    passages = list(grouped_df.iloc)
    X = np.array([sent2features(s) for s in passages])
    y = np.array([sent2labels(s) for s in passages])
    return X, y


def word2features(sent, i: int) -> dict:
    word = str(sent[i][0])
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-5:]": word[-5:],
        "word[-4:]": word[-4:],
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[-1:]": word[-1:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }

    if i == 0:
        features["BOS"] = True
    if i == len(sent):
        features["EOS"] = True

    for step in (1, 2, 3, 4, 5):
        if i > step - 1:
            word1 = str(sent[i - step][0])
            postag1 = sent[i - step][1]
            features.update(
                {
                    f"-{step}:word.lower()": word1.lower(),
                    f"-{step}:word.istitle()": word1.istitle(),
                    f"-{step}:word.isupper()": word1.isupper(),
                    f"-{step}:postag": postag1,
                    f"-{step}:postag[:2]": postag1[:2],
                }
            )

        if i < len(sent) - step:
            word1 = str(sent[i + step][0])
            postag1 = sent[i + step][1]
            features.update(
                {
                    f"+{step}:word.lower()": word1.lower(),
                    f"+{step}:word.istitle()": word1.istitle(),
                    f"+{step}:word.isupper()": word1.isupper(),
                    f"+{step}:postag": postag1,
                    f"+{step}:postag[:2]": postag1[:2],
                }
            )

    return features


def sent2features(sent) -> List[dict]:
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent) -> List[str]:
    return [label for token, postag, label in sent]


def sent2tokens(sent) -> List[str]:
    return [token for token, postag, label in sent]


def evaluate(crf, X_test, y_test, report_output_file_name):
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)
    labels.remove("O")
    report_output_file_name.write_text(
        crf_metrics.flat_classification_report(y_test, y_pred, labels=labels)
    )


if __name__ == "__main__":
    plac.call(main)
