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

import numpy as np
import pandas as pd

from mostlyai.qa import _embeddings


def test_string_columns_share_one_embedder(monkeypatch):
    load = Mock()
    load.return_value.encode.side_effect = lambda values: np.random.default_rng(0).normal(size=(len(values), 8))
    monkeypatch.setattr(_embeddings, "load_embedder", load)
    data = pd.DataFrame({"a": ["one", "two", "three"], "b": ["four", "five", "six"]})
    syn, trn, hol = _embeddings.encode_strings(data, data, data)
    load.assert_called_once_with()
    assert load.return_value.encode.call_count == 2
    assert syn.shape == trn.shape == hol.shape == (3, 4)
    pd.testing.assert_frame_equal(syn, trn)
    pd.testing.assert_frame_equal(trn, hol)


def test_no_string_columns_do_not_load_embedder(monkeypatch):
    load = Mock()
    monkeypatch.setattr(_embeddings, "load_embedder", load)
    data = pd.DataFrame(index=range(3))
    syn, trn, hol = _embeddings.encode_strings(data, data)
    load.assert_not_called()
    assert syn.shape == trn.shape == (3, 0)
    assert hol is None
