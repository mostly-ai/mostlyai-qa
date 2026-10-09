# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import pandas as pd
from joblib import Parallel

from mostlyai.qa import _accuracy


def test_bivariate_counts_compute_each_pair_once(monkeypatch):
    binned = pd.DataFrame(
        {
            "tgt::a": pd.Categorical(["z", "a", "z", "a"], categories=["z", "a", "unused"], ordered=True),
            "tgt::b": pd.Categorical([2, 1, 1, 1], categories=[2, 1, 3], ordered=True),
            "ctx::c": pd.Categorical(["x", "x", "y", "y"]),
            "nxt::a": pd.Categorical(["a", "z", "a", "z"]),
        }
    )
    original = _accuracy.bin_count_biv
    counted = Mock(wraps=original)
    monkeypatch.setattr(_accuracy, "bin_count_biv", counted)
    # Keep the spy in this process.
    monkeypatch.setattr(_accuracy, "Parallel", lambda: Parallel(n_jobs=1))
    _, counts = _accuracy.calculate_bin_counts(binned)
    pairs = _accuracy.calculate_bivariate_columns(binned, append_symetric=False)
    assert counted.call_count == len(pairs)
    assert len(counts) == 2 * len(pairs)
    for row in _accuracy.calculate_bivariate_columns(binned).itertuples():
        _, expected = original(row.col1, row.col2, binned[row.col1], binned[row.col2])
        pd.testing.assert_series_equal(counts[(row.col1, row.col2)], expected)
