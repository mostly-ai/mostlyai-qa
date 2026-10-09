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

import warnings

import pandas as pd
import pytest

from mostlyai.qa._accuracy import bin_categorical
from mostlyai.qa._common import EMPTY_BIN, NA_BIN, OTHER_BIN, RARE_BIN


@pytest.mark.parametrize("bins", [["keep"], 1])
def test_unknown_categories_are_mapped_without_warnings(bins):
    data = pd.Series(["keep"] * 5 + ["unknown", None, "", RARE_BIN])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        categorical, categories = bin_categorical(data, bins)
    assert list(categorical) == ["keep"] * 5 + [OTHER_BIN, NA_BIN, EMPTY_BIN, RARE_BIN]
    assert categories == ["keep", EMPTY_BIN, OTHER_BIN, RARE_BIN, NA_BIN]
    assert categorical.ordered
