from pathlib import Path

import pandas as pd

from src.definitions import PROJECT_ROOT
from src.preprocess_data import preprocess_data


def test_preprocess_data(tmp_path: Path):
    preprocess_data(
        data_root=PROJECT_ROOT / "data/test/raw",
        output_dir=tmp_path,
    )

    pd.testing.assert_frame_equal(
        left=pd.read_csv(tmp_path / "labeled_passages.csv"),
        right=pd.read_csv(PROJECT_ROOT / "data/test/expected_labeled_passages.csv"),
    )
