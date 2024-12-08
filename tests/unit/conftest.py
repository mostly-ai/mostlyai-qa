# Copyright 2024 MOSTLY AI
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

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from mostlyai.qa._filesystem import TemporaryWorkspace


@pytest.fixture()
def cat_col():
    trn_values = ["cat_0"] * 5 + [f"cat_{x}" for x in range(5, 90)] + ["cat_90"] * 10
    syn_values = [f"cat_{x}" for x in range(50, 150)]
    yield pd.DataFrame({"cat": trn_values}), pd.DataFrame({"cat": syn_values})


@pytest.fixture()
def cat_num_col():
    trn_values = np.repeat(range(5), 20)
    syn_values = [x for x in range(1, 101)]
    yield pd.DataFrame({"cat_num": trn_values}), pd.DataFrame({"cat_num": syn_values})


@pytest.fixture()
def num_col():
    trn_values = [0] * 5 + [x for x in range(30, 80)] + [80] * 45
    syn_values = [x for x in range(50, 150)]
    yield pd.DataFrame({"num": trn_values}), pd.DataFrame({"num": syn_values})


@pytest.fixture()
def cat_dt_col():
    trn_values = np.repeat(pd.date_range(start="2018-01-01", periods=5, freq="D"), 20)
    syn_values = pd.date_range(start="2018-01-02", periods=100, freq="D")
    yield pd.DataFrame({"cat_dt": trn_values}), pd.DataFrame({"cat_dt": syn_values})


@pytest.fixture()
def dt_col():
    date_1 = pd.to_datetime("2018-01-01")
    date_2 = date_1 + timedelta(35)
    date_3 = date_2 + timedelta(20)
    trn_values = [date_1] * 35 + list(pd.date_range(start=date_2, periods=20, freq="D")) + [date_3] * 45
    syn_values = pd.date_range(start=date_1 + timedelta(40), periods=100, freq="D")
    yield pd.DataFrame({"dt": trn_values}), pd.DataFrame({"dt": syn_values})


@pytest.fixture()
def cols(cat_col, num_col, dt_col):
    trn_syn_cols = cat_col, num_col, dt_col
    trn_cols, syn_cols = zip(*trn_syn_cols)
    return pd.concat(trn_cols, axis=1), pd.concat(trn_cols, axis=1), pd.concat(syn_cols, axis=1)


@pytest.fixture()
def cols_5(cat_col, cat_num_col, num_col, cat_dt_col, dt_col):
    trn_syn_cols = cat_col, cat_num_col, num_col, cat_dt_col, dt_col
    trn_cols, syn_cols = zip(*trn_syn_cols)
    return pd.concat(trn_cols, axis=1), pd.concat(trn_cols, axis=1), pd.concat(syn_cols, axis=1)


@pytest.fixture
def workspace():
    with TemporaryWorkspace() as ws:
        yield ws
