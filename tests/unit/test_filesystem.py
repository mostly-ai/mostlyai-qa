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

from mostlyai.qa._accuracy import (
    bin_data,
    calculate_bin_counts,
    calculate_categorical_uni_counts,
)
from mostlyai.qa._filesystem import Statistics


class TestStatistics:
    def test_store_and_load_categorical_uni_counts(self, tmp_path, cols):
        # this tests covers two symetrical functions:
        # store_categorical_uni_counts and load_categorical_uni_counts
        trn, _, _ = cols
        # apply ctx:/tgt: prefixes and create nxt: columns
        prefixes = ["ctx:", "ctx:", "tgt:"]
        columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
        trn.columns = columns
        trn["nxt:dt"] = trn["tgt:dt"]
        uni_counts = calculate_categorical_uni_counts(trn)

        statistics = Statistics(path=tmp_path)
        statistics.store_categorical_uni_counts(uni_counts)
        uni_counts_loaded = statistics.load_categorical_uni_counts()

        assert uni_counts.keys() == uni_counts_loaded.keys()
        for uni_bin_counts_col, uni_bin_counts_loaded_col in zip(uni_counts.values(), uni_counts_loaded.values()):
            pd.testing.assert_series_equal(uni_bin_counts_col, uni_bin_counts_loaded_col)

    def test_store_and_load_bin_counts(self, tmp_path, cols):
        # this tests covers two symetrical functions:
        # store_bin_counts and load_bin_counts
        trn, hol, syn = cols
        # apply ctx:/tgt: prefixes and create nxt: columns
        prefixes = ["ctx:", "ctx:", "tgt:"]
        columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
        trn.columns, syn.columns = columns, columns
        trn["nxt:dt"], syn["nxt:dt"] = trn["tgt:dt"], syn["tgt:dt"]
        trn, bins = bin_data(trn, 3)
        syn, _ = bin_data(syn, bins)
        uni_bin_counts, biv_bin_counts = calculate_bin_counts(trn)

        statistics = Statistics(path=tmp_path)
        statistics.store_bin_counts(uni_bin_counts, biv_bin_counts)
        (
            uni_bin_counts_loaded,
            biv_bin_counts_loaded,
        ) = statistics.load_bin_counts()

        assert uni_bin_counts.keys() == uni_bin_counts_loaded.keys()
        for uni_bin_counts_col, uni_bin_counts_loaded_col in zip(
            uni_bin_counts.values(), uni_bin_counts_loaded.values()
        ):
            pd.testing.assert_series_equal(uni_bin_counts_col, uni_bin_counts_loaded_col, check_names=False)

        assert biv_bin_counts.keys() == biv_bin_counts_loaded.keys()
        for biv_bin_counts_col, biv_bin_counts_loaded_col in zip(
            biv_bin_counts.values(), biv_bin_counts_loaded.values()
        ):
            pd.testing.assert_series_equal(
                biv_bin_counts_col,
                biv_bin_counts_loaded_col,
                check_index=False,
                check_names=False,
            )
