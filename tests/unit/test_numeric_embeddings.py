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

import numpy as np
import pandas as pd
import pytest

from mostlyai.qa._embeddings import encode_numerics


@pytest.mark.parametrize("with_holdout", [False, True])
def test_synthetic_missing_values_are_distinct_from_median(with_holdout):
    trn = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]})
    syn = pd.DataFrame({"value": [np.nan, 2.0]})
    hol = trn.copy() if with_holdout else None
    syn_encoded, trn_encoded, hol_encoded = encode_numerics(syn, trn, hol)
    assert syn_encoded["value"].tolist() == [0.0, 0.0]
    assert syn_encoded["value - N/A"].tolist() == [0.5, -0.5]
    assert (trn_encoded["value - N/A"] == -0.5).all()
    if with_holdout:
        assert (hol_encoded["value - N/A"] == -0.5).all()
    else:
        assert hol_encoded is None


def test_complete_numeric_values_keep_existing_dimensions():
    data = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]})
    syn_encoded, trn_encoded, _ = encode_numerics(data, data)
    assert list(syn_encoded.columns) == list(trn_encoded.columns) == ["value"]
