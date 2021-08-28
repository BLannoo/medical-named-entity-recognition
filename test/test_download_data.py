import json
from typing import List, Tuple

import assertpy
import pytest

from download_data import generate_batches, download_batch


@pytest.mark.parametrize(
    "last_id,batches",
    [
        (1000, [(1, 1000)]),
        (500, [(1, 500)]),
        (2000, [(1, 1000), (1001, 2000)]),
        (1500, [(1, 1000), (1001, 1500)]),
    ],
)
def test_generate_batches(last_id: int, batches: List[Tuple[int, int]]):
    assertpy.assert_that(
        generate_batches(last_id),
    ).contains_only(*batches)


def test_download_batch():
    assertpy.assert_that(
        [
            json.loads(example) if example else example
            for example in download_batch(1, 1000).split("\n")
        ]
    ).is_length(1000).extracting("id").contains(
        "1",
        "1000",
    )
