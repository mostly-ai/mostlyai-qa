# Copyright 2024-2025 MOSTLY AI
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

import pandas as pd
import numpy as np

from mostlyai.qa._sampling import pull_data_for_embeddings


def test_pull_data_for_embeddings_groupby(tmp_path):
    df = pd.DataFrame({"id": ["a", "b", "a", "a"], "x": ["a0", "b0", "a1", "a2"], "y": ["a2", "b0", "a1", "a0"]})
    strings = pull_data_for_embeddings(df_tgt=df, tgt_context_key="id")
    assert "b0 b0" in strings
    assert "a0 a2, a1 a1, a2 a0" in strings
    assert len(strings) == df.id.nunique()


def test_pull_data_for_embeddings_large_int(tmp_path):
    # regression test for issue with overly large integers
    df = pd.DataFrame(
        {"cc": list(np.random.randint(100, 200, size=1000)) + [1800218404984585216] + [pd.NA]}, dtype="Int64"
    )
    pull_data_for_embeddings(df_tgt=df)


def test_pull_data_for_embeddings_dates(tmp_path):
    n = 1000
    dates = pd.to_datetime(np.random.randint(pd.Timestamp("2020-01-01").value, pd.Timestamp("2025-01-01").value, n))
    df = pd.DataFrame({"x": dates, "y": dates, "z": dates})
    df["y"] = df["y"].values.astype("datetime64[s]")
    df["z"] = df["z"].values.astype("datetime64[D]")
    df.loc[0] = pd.NaT
    pull_data_for_embeddings(df_tgt=df)
