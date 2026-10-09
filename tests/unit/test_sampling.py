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

from mostlyai.qa import _sampling


@pytest.mark.parametrize("with_holdout", [False, True])
def test_sequence_embeddings_preserve_inputs(monkeypatch, with_holdout):
    frames = [pd.DataFrame({"key": [0, 0, 1, 1], "value": [1, 2, 3, 4]}) for _ in range(3)]
    originals = [frame.copy(deep=True) for frame in frames]
    monkeypatch.setattr(
        _sampling,
        "encode_data",
        lambda syn_data, trn_data, hol_data: (
            np.zeros((len(syn_data), 1)),
            np.zeros((len(trn_data), 1)),
            np.zeros((len(hol_data), 1)) if hol_data is not None else None,
        ),
    )
    for _ in range(2):
        _sampling.prepare_data_for_embeddings(
            syn_tgt_data=frames[0],
            trn_tgt_data=frames[1],
            hol_tgt_data=frames[2] if with_holdout else None,
            tgt_context_key="key",
            max_sample_size=100,
        )
        for frame, original in zip(frames, originals):
            pd.testing.assert_frame_equal(frame, original)
