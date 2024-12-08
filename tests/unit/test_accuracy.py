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

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from mostlyai.qa._accuracy import (
    calculate_accuracy,
    calculate_bivariate_columns,
    calculate_expected_l1_multinomial,
    plot_univariate_distribution_numeric,
    calculate_bin_counts,
    calculate_bivariates,
    calculate_categorical_uni_counts,
    calculate_numeric_uni_kdes,
    calculate_univariates,
    plot_store_accuracy_matrix,
    plot_store_bivariates,
    plot_store_univariates,
    bin_data,
    bin_numeric,
    bin_datetime,
    search_bin_boundaries,
    trim_labels,
    calculate_correlations,
    plot_store_correlation_matrices,
    bin_categorical,
)
from mostlyai.qa._sampling import pull_data_for_accuracy, sample_two_consecutive_rows
from mostlyai.qa._common import (
    OTHER_BIN,
    EMPTY_BIN,
    NA_BIN,
    RARE_BIN,
    CTX_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
    NXT_COLUMN_PREFIX,
)


def test_calculate_univariates(cols):
    trn, hol, syn = cols
    # prefix some columns with "tgt::"
    columns = [f"tgt::{c}" if idx > 0 else c for idx, c in enumerate(trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    uni_acc = calculate_univariates(trn, syn)
    assert uni_acc.columns.to_list() == ["column", "accuracy", "accuracy_max"]
    assert (uni_acc["accuracy"] >= 0.5).all()
    assert (uni_acc["accuracy_max"] >= 0.5).all()
    assert uni_acc.shape[0] == len([c for c in trn if c.startswith("tgt::")])


def test_calculate_bivariates_columns(cols):
    trn, _, _ = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "_::", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns = columns
    trn["nxt::dt"] = trn["tgt::dt"]
    columns_df = calculate_bivariate_columns(trn)
    assert columns_df.columns.to_list() == [
        "col1",
        "col2",
        "type",
    ]
    assert columns_df.shape[0] == 4  # tgt vs ctx, tgt vs nxt - symetric


def test_calculate_bivariates(cols):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "_.", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    biv_acc = calculate_bivariates(trn, syn)
    assert biv_acc.columns.to_list() == [
        "col1",
        "col2",
        "type",
        "accuracy",
        "accuracy_max",
    ]
    assert (biv_acc["accuracy"] >= 0.5).all()
    assert (biv_acc["accuracy_max"] >= 0.5).all()
    assert biv_acc.shape[0] == trn.shape[1]


def test_calculate_uni_counts(tmp_path, cols):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "ctx::", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    syn = syn[[c for c in syn.columns if c != "ctx::num"]]
    uni_counts_trn = calculate_categorical_uni_counts(trn, hash_rare_values=True)
    uni_counts_syn = calculate_categorical_uni_counts(syn, uni_counts_trn, hash_rare_values=False)

    # only categorical columns are considered
    assert set(uni_counts_trn.keys()) == {"ctx::cat"}
    assert set(uni_counts_syn.keys()) == {"ctx::cat"}

    assert uni_counts_trn["ctx::cat"].sum() == 100
    assert uni_counts_syn["ctx::cat"].sum() == 100

    assert uni_counts_trn["ctx::cat"].size == 87
    assert uni_counts_syn["ctx::cat"].size == 100

    # all the values are protected, but cat_90 (it appears 10 times)
    assert set(trn["ctx::cat"].values).intersection(set(uni_counts_trn["ctx::cat"].keys())) == {"cat_90"}
    # none of the values are protected (it's synthetic data)
    assert set(trn["ctx::cat"].values).intersection(set(uni_counts_syn["ctx::cat"].keys())) == {
        f"cat_{i}" for i in range(50, 91)
    }
    # thus, only one category appear both in trn and syn counts
    assert len(set(uni_counts_trn["ctx::cat"].keys()).intersection(set(uni_counts_syn["ctx::cat"].keys()))) == 1
    # protected values have the following pattern: "rare_<hash>"
    assert uni_counts_trn["ctx::cat"].keys()[0].startswith("rare_")


def test_calculate_bin_counts(tmp_path, cols):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "ctx::", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    uni_bin_counts, biv_bin_counts = calculate_bin_counts(trn)

    assert set(uni_bin_counts.keys()) == {"ctx::cat", "ctx::num", "tgt::dt", "nxt::dt"}
    assert uni_bin_counts["ctx::cat"].to_dict() == {
        "(other)": 85,
        "cat_0": 5,
        "cat_90": 10,
    }
    assert uni_bin_counts["ctx::num"].to_dict() == {"⪰ 0": 24, "⪰ 49": 25, "⪰ 74": 51}
    assert uni_bin_counts["tgt::dt"].to_dict() == {
        "⪰ 2018-Feb-07": 12,
        "⪰ 2018-Feb-19": 51,
        "⪰ 2018-Jan-01": 37,
    }
    assert uni_bin_counts["nxt::dt"].to_dict() == {
        "⪰ 2018-Feb-07": 12,
        "⪰ 2018-Feb-19": 51,
        "⪰ 2018-Jan-01": 37,
    }

    assert set(biv_bin_counts.keys()) == {
        ("tgt::dt", "ctx::num"),
        ("ctx::cat", "tgt::dt"),
        ("nxt::dt", "tgt::dt"),
        ("tgt::dt", "ctx::cat"),
        ("ctx::num", "tgt::dt"),
        ("tgt::dt", "nxt::dt"),
    }
    assert biv_bin_counts[("tgt::dt", "ctx::num")].to_dict() == {
        ("⪰ 2018-Feb-07", "⪰ 49"): 12,
        ("⪰ 2018-Feb-19", "⪰ 74"): 51,
        ("⪰ 2018-Jan-01", "⪰ 0"): 24,
        ("⪰ 2018-Jan-01", "⪰ 49"): 13,
    }
    assert biv_bin_counts[("ctx::cat", "tgt::dt")].to_dict() == {
        ("(other)", "⪰ 2018-Feb-07"): 12,
        ("(other)", "⪰ 2018-Feb-19"): 41,
        ("(other)", "⪰ 2018-Jan-01"): 32,
        ("cat_0", "⪰ 2018-Jan-01"): 5,
        ("cat_90", "⪰ 2018-Feb-19"): 10,
    }
    assert biv_bin_counts[("nxt::dt", "tgt::dt")].to_dict() == {
        ("⪰ 2018-Feb-07", "⪰ 2018-Feb-07"): 12,
        ("⪰ 2018-Feb-19", "⪰ 2018-Feb-19"): 51,
        ("⪰ 2018-Jan-01", "⪰ 2018-Jan-01"): 37,
    }
    assert biv_bin_counts[("tgt::dt", "ctx::cat")].to_dict() == {
        ("⪰ 2018-Feb-07", "(other)"): 12,
        ("⪰ 2018-Feb-19", "(other)"): 41,
        ("⪰ 2018-Feb-19", "cat_90"): 10,
        ("⪰ 2018-Jan-01", "(other)"): 32,
        ("⪰ 2018-Jan-01", "cat_0"): 5,
    }
    assert biv_bin_counts[("ctx::num", "tgt::dt")].to_dict() == {
        ("⪰ 0", "⪰ 2018-Jan-01"): 24,
        ("⪰ 49", "⪰ 2018-Feb-07"): 12,
        ("⪰ 49", "⪰ 2018-Jan-01"): 13,
        ("⪰ 74", "⪰ 2018-Feb-19"): 51,
    }
    assert biv_bin_counts[("tgt::dt", "nxt::dt")].to_dict() == {
        ("⪰ 2018-Feb-07", "⪰ 2018-Feb-07"): 12,
        ("⪰ 2018-Feb-19", "⪰ 2018-Feb-19"): 51,
        ("⪰ 2018-Jan-01", "⪰ 2018-Jan-01"): 37,
    }


def test_plot_store_univariates(cols, workspace):
    trn, hol, syn = cols
    trn = trn.rename(columns=lambda c: f"tgt::{c}")
    syn = syn.rename(columns=lambda c: f"tgt::{c}")
    trn_3, bins = bin_data(trn, 3)
    syn_3, _ = bin_data(syn, bins)
    trn_kdes = calculate_numeric_uni_kdes(trn)
    syn_kdes = calculate_numeric_uni_kdes(syn, trn_kdes)
    trn_cat_cnts = calculate_categorical_uni_counts(trn)
    syn_cat_cnts = calculate_categorical_uni_counts(syn, trn_cat_cnts)
    uni_acc = calculate_univariates(trn_3, syn_3)
    trn_3_cnts, _ = calculate_bin_counts(trn_3)
    syn_3_cnts, _ = calculate_bin_counts(syn_3)
    plot_store_univariates(
        trn_kdes,
        syn_kdes,
        trn_cat_cnts,
        syn_cat_cnts,
        trn_3_cnts,
        syn_3_cnts,
        uni_acc,
        workspace,
        show_accuracy=True,
    )
    output_dir = workspace.workspace_dir / "figures" / "univariate"
    assert len(list(output_dir.glob("*.html"))) == 3


def test_plot_store_bivariates(cols, workspace):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "tgt::", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::num"], syn["nxt::num"] = trn["tgt::num"], syn["tgt::num"]
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    trn_cnts_uni, trn_cnts_biv = calculate_bin_counts(trn)
    syn_cnts_uni, syn_cnts_biv = calculate_bin_counts(syn)
    acc_biv = calculate_bivariates(trn, syn)
    plot_store_bivariates(
        trn_cnts_uni,
        syn_cnts_uni,
        trn_cnts_biv,
        syn_cnts_biv,
        acc_biv=acc_biv,
        workspace=workspace,
        show_accuracy=True,
    )
    output_dir = workspace.workspace_dir / "figures" / "bivariate"
    assert len(list(output_dir.glob("*.html"))) == 10


def test_plot_store_accuracy_matrix(cols, workspace):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "_::", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    uni_acc = calculate_univariates(trn, syn)
    biv_acc = calculate_bivariates(trn, syn)
    plot_store_accuracy_matrix(uni_acc, biv_acc, workspace)
    output_dir = workspace.workspace_dir / "figures"
    assert len(list(output_dir.glob("*.html"))) == 1


def test_calculate_accuracy():
    trn_bin_cols = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    syn_bin_cols = pd.DataFrame({"x": ["a", "a", "c", "c"]})
    observed_acc, _ = calculate_accuracy(trn_bin_cols, syn_bin_cols)
    assert observed_acc == 0.5

    probs = [0.5, 0.2, 0.2, 0.1]
    n_1 = 6_000
    n_2 = 12_000
    vals = [i for i in range(len(probs))]
    trn_bin_cols = pd.DataFrame({"x": np.random.choice(vals, n_1, p=probs)})
    syn_bin_cols = pd.DataFrame({"x": np.random.choice(vals, n_2, p=probs)})
    observed_acc, expected_acc = calculate_accuracy(trn_bin_cols, syn_bin_cols)
    assert observed_acc > 0.9
    assert expected_acc > 0.9
    assert np.abs(observed_acc - expected_acc) < 0.018


def test_calculate_expected_l1_multinomial():
    # test implemented exact formula against bootstrapped results
    def bootstrap_expected_l1_multinomial(probs, n_1, n_2):
        def _bootstrap_l1(probs, n_1, n_2):
            vals = [i for i in range(len(probs))]
            v1 = pd.Series(np.random.choice(vals, n_1, p=probs)).value_counts(normalize=True)
            v2 = pd.Series(np.random.choice(vals, n_2, p=probs)).value_counts(normalize=True)
            return (v1 - v2).abs().sum()

        boot_reps = 1_000
        return np.mean([_bootstrap_l1(probs, n_1, n_2) for i in range(boot_reps)])

    probs = [0.5, 0.2, 0.2, 0.1]
    n1 = 3_000
    n2 = 6_000
    exact_l1 = calculate_expected_l1_multinomial(probs, n1, n2)
    boots_l1 = bootstrap_expected_l1_multinomial(probs, n1, n2)
    assert np.abs(exact_l1 - boots_l1) < 0.01

    probs = [0.9, 0.09, 0.01]
    n1 = 8_000
    n2 = 2_000
    exact_l1 = calculate_expected_l1_multinomial(probs, n1, n2)
    boots_l1 = bootstrap_expected_l1_multinomial(probs, n1, n2)
    assert np.abs(exact_l1 - boots_l1) < 0.01


def test_plot_univariate_distribution_numeric():
    # regression test for tiny constants
    trn_col = syn_col = pd.Series([1e-12] * 10_000 + [np.nan], dtype="Float64")
    plot_univariate_distribution_numeric(trn_col, syn_col)
    # regression test for large constants
    trn_col = pd.Series([1596570000000] * 10_000 + [np.nan], dtype="Int64")
    syn_col = pd.Series([1596570000001] * 10_000 + [np.nan], dtype="Int64")
    plot_univariate_distribution_numeric(trn_col, syn_col)
    # regression test for one non-na value
    trn_col = syn_col = pd.Series([1234.56] + [np.nan] * 10_000, dtype="Float64")
    plot_univariate_distribution_numeric(trn_col, syn_col)


def test_sample_two_consecutive_rows():
    df = pd.DataFrame({"id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4], "x": [1, 2, 3, 5, 1, 2, 3, 1, 2, 1]})
    first_rows, second_rows = sample_two_consecutive_rows(df=df, col_by="id")
    assert len(first_rows) == 4
    assert len(second_rows) == 3
    assert (first_rows["x"][0:2] == second_rows["x"][0:2] - 1).all()


class TestPrepareAccuracyData:
    def test_tgt(self):
        df_tgt = pd.DataFrame(
            {
                "col1": np.random.choice(a=["a", "b", "c"], size=100),
                "col2": np.random.randint(low=0, high=100, size=100),
            }
        )
        df = pull_data_for_accuracy(df_tgt=df_tgt)
        assert df.shape[0] == len(df_tgt)
        assert list(df.columns) == [f"{TGT_COLUMN_PREFIX}col1", f"{TGT_COLUMN_PREFIX}col2"]

    def test_explicit_ctx_tgt(self):
        df_ctx = pd.DataFrame(
            {
                "pk": list(range(50)),
                "col1": np.random.choice(a=["x", "y", "z"], size=50),
            }
        )
        df_tgt = pd.DataFrame(
            {
                "fk": list(range(50)) * 2,
                "col1": np.random.choice(a=["a", "b", "c"], size=100),
                "col2": np.random.randint(low=0, high=10, size=100),
            }
        )
        df = pull_data_for_accuracy(
            df_ctx=df_ctx,
            df_tgt=df_tgt,
            ctx_primary_key="pk",
            tgt_context_key="fk",
        )
        assert df.shape[0] == len(df_ctx)
        assert list(df.columns) == [
            f"{CTX_COLUMN_PREFIX}col1",
            f"{TGT_COLUMN_PREFIX}Sequence Length",
            f"{TGT_COLUMN_PREFIX}col1",
            f"{TGT_COLUMN_PREFIX}col2",
            f"{NXT_COLUMN_PREFIX}col1",
            f"{NXT_COLUMN_PREFIX}col2",
        ]

    def test_implicit_ctx_tgt(self):
        df_tgt = pd.DataFrame(
            {
                "fk": list(range(50)) * 2,
                "col1": np.random.choice(a=["a", "b", "c"], size=100),
                "col2": np.random.randint(low=0, high=10, size=100),
            }
        )
        df = pull_data_for_accuracy(df_tgt=df_tgt, tgt_context_key="fk")
        assert df.shape[0] == 50
        assert list(df.columns) == [
            f"{TGT_COLUMN_PREFIX}Sequence Length",
            f"{TGT_COLUMN_PREFIX}col1",
            f"{TGT_COLUMN_PREFIX}col2",
            f"{NXT_COLUMN_PREFIX}col1",
            f"{NXT_COLUMN_PREFIX}col2",
        ]


class TestBinData:
    def test_cat_col(self, cat_col):
        trn, syn = cat_col
        trn, bins = bin_data(trn, 5)
        syn, _ = bin_data(syn, bins)

        # though 5 bins are requested, only 2 qualifies
        # due to rare category protection
        trn_counts = trn["cat"].value_counts().to_dict()
        assert trn_counts == {
            "cat_0": 5,
            "cat_90": 10,
            OTHER_BIN: 85,
            EMPTY_BIN: 0,
            NA_BIN: 0,
            RARE_BIN: 0,
        }

        syn_counts = syn["cat"].value_counts().to_dict()
        assert syn_counts == {
            "cat_0": 0,
            "cat_90": 1,
            OTHER_BIN: 99,
            EMPTY_BIN: 0,
            NA_BIN: 0,
            RARE_BIN: 0,
        }

    def test_num_col(self, num_col):
        trn, syn = num_col
        trn, bins = bin_data(trn, 2)
        syn, _ = bin_data(syn, bins)

        trn_counts = trn["num"].value_counts().to_dict()
        assert trn_counts == {
            "⪰ 0": 49,
            "⪰ 74": 51,
            NA_BIN: 0,
        }

        syn_counts = syn["num"].value_counts().to_dict()
        assert syn_counts == {
            "⪰ 0": 24,
            "⪰ 74": 76,
            NA_BIN: 0,
        }

    def test_dt_col(self, dt_col):
        trn, syn = dt_col
        trn, bins = bin_data(trn, 2)
        syn, _ = bin_data(syn, bins)

        trn_counts = trn["dt"].value_counts().to_dict()
        assert trn_counts == {"⪰ 2018-Feb-19": 51, "⪰ 2018-Jan-01": 49, NA_BIN: 0}

        syn_counts = syn["dt"].value_counts().to_dict()
        assert syn_counts == {"⪰ 2018-Feb-19": 91, "⪰ 2018-Jan-01": 9, NA_BIN: 0}

    def test_num_col_precisions(self):
        # test 0.01 bin has only 0.01 values
        df = pd.DataFrame({"num": [0.01] * 5 + [0.02] * 5 + [0.99] * 5})
        df, _ = bin_data(df, 2)
        df_counts = df["num"].value_counts().to_dict()
        assert df_counts == {"(n/a)": 0, "⪰ 0.01": 5, "⪰ 0.02": 10}

        # test 0.0 bin has only 0.01 values
        df = pd.DataFrame({"num": [0.01] * 5 + [0.99] * 5 + [1.99] * 5})
        df, _ = bin_data(df, 2)
        df_counts = df["num"].value_counts().to_dict()
        assert df_counts == {"(n/a)": 0, "⪰ 0.0": 5, "⪰ 0.9": 10}

        # test each bin has at least 2 unique values
        df = pd.DataFrame({"num": [0.01] * 5 + [0.88] * 5 + [0.98] * 5 + [0.99] * 5 + [1.02] * 5 + [1.99] * 5})
        df, _ = bin_data(df, 2)
        df_counts = df["num"].value_counts().to_dict()
        assert df_counts == {"(n/a)": 0, "⪰ 0.0": 10, "⪰ 0.9": 20}

        # test very small precisions
        df = pd.DataFrame({"num": [1e-12] * 5 + [2e-12] * 5 + [9.9e-12] * 5 + [1 + 9.9e-12] * 5})
        df, _ = bin_data(df, 2)
        df_counts = df["num"].value_counts().to_dict()
        assert df_counts == {"(n/a)": 0, "⪰ 1e-12": 5, "⪰ 2e-12": 15}

    def test_num_col_nans_only(self):
        df = pd.DataFrame({"nans": [float("nan")] * 10})
        df, _ = bin_data(df, 2)
        df_counts = df["nans"].value_counts().to_dict()
        assert df_counts["(n/a)"] == 10

    def test_bin_categorical(self):
        x = pd.Series(["a", "b"] * 50 + ["x"])
        col, _ = bin_categorical(x, 5)
        assert len(col) == 101
        x = pd.Series([True, False] * 50 + [np.nan] * 100, dtype="object")
        col, _ = bin_categorical(x, 5)
        assert len(col) == 200

    def test_bin_numeric(self):
        # test several edge cases
        cases = [
            (pd.Series([pd.NA], dtype="Int64"), ["(n/a)"]),
            (pd.Series([pd.NaT] * 5 + [pd.NA] * 5, dtype="object"), ["(n/a)"] * 10),
            (
                pd.Series([10000000000000000, 10000000000000005], dtype="Float64"),
                ["⪰ 10000000000000000", "(n/a)"],
            ),
            (pd.Series([10] * 10), ["⪰ 10"] * 10),  # single value
            (
                pd.Series([10] * 10 + [30] * 10),
                ["⪰ 10"] * 10 + ["⪰ 30"] * 10,
            ),  # two values
        ]

        for col, expected in cases:
            col, _ = bin_numeric(col, 10)
            assert col.tolist() == expected

    def test_bin_datetime(self):
        # test several edge cases
        cases = [
            (pd.Series([pd.NaT] * 5, dtype="datetime64[ns]"), ["(n/a)"] * 5),
            (
                pd.Series([pd.Timestamp.max] * 10 + [pd.to_datetime("1920-12-20")] * 10),
                ["(n/a)"] * 10 + ["⪰ 1920"] * 10,
            ),
            (
                pd.Series([pd.to_datetime("1920-12-20")] * 10, dtype="datetime64[ns]"),
                ["⪰ 1920-Dec-20"] * 10,
            ),  # single value
            (
                pd.Series(
                    [pd.to_datetime("2023-01-30 13:00:00.333")] * 10 + [pd.to_datetime("2023-01-30 13:00:00.334")] * 10
                ),
                ["⪰ 2023-01-30 13:00:00.333000"] * 20,
            ),  # two values
            (
                pd.Series([pd.NaT, "2024-11-20"], dtype="datetime64[ns]"),
                ["(n/a)", "⪰ 2024-Nov-20"],
            ),  # single value with leading N/A
        ]

        for col, expected in cases:
            col, _ = bin_datetime(col, 10)
            assert col.tolist() == expected

    def test_cols_with_a_slight_difference_in_types(self):
        trn = pd.DataFrame({"num": [0.0] * 2 + [0.5] * 5 + [1.0] * 3}, dtype=np.float32)
        syn = pd.DataFrame({"num": [0.0] * 5 + [0.5] * 9 + [1.0] * 8}, dtype=np.float64)
        trn_bin, bins = bin_data(trn, 2)
        syn_bin, _ = bin_data(syn, bins)
        trn_counts = trn_bin["num"].value_counts().to_dict()
        syn_counts = syn_bin["num"].value_counts().to_dict()
        assert trn_counts == {"⪰ 0.5": 8, "⪰ 0.0": 2, "(n/a)": 0}
        assert syn_counts == {"⪰ 0.5": 17, "⪰ 0.0": 5, "(n/a)": 0}


def test_search_bin_boundaries(num_col):
    trn_col = num_col[0]["num"]

    # fix input, n = 1 - 5
    exps = [
        [0, 80],
        [0, 74, 80],
        [0, 49, 74, 80],
        [0, 41, 58, 74, 80],
        [0, 37, 49, 62, 74, 80],
    ]
    for n, exp in enumerate(exps, start=1):
        assert_array_equal(search_bin_boundaries(trn_col, n), exp, err_msg=f"n={n}")

    # fix input, n > len(trn_col)
    assert_array_equal(search_bin_boundaries(trn_col, len(trn_col) + 10), trn_col.unique())

    # fix n=3, input growing
    input_exp = [
        (pd.Series(np.linspace(1, 4, 4)), [1, 2, 3, 4]),
        (pd.Series(np.linspace(1, 5, 5)), [1, 2, 3, 5]),
        (pd.Series(np.linspace(1, 6, 6)), [1, 2, 4, 6]),
        (pd.Series(np.linspace(1, 7, 7)), [1, 3, 5, 7]),
    ]
    for n_trn_col, (trn_col, exp) in enumerate(input_exp, start=1):
        assert_array_equal(search_bin_boundaries(trn_col, 3), exp, err_msg=f"n_trn_col={n_trn_col}")


def test_trim_labels():
    labels = ["a" * 2 + x + "a" * 2 for x in ["aa", "ab", "ba", "bb"]]
    assert trim_labels(labels, max_length=5, ensure_unique=False) == [
        "a...a",
        "a...a",
        "a...a",
        "a...a",
    ]
    assert trim_labels(labels, max_length=5, ensure_unique=True) == [
        "a...a",
        "a...a0",
        "a...a1",
        "a...a2",
    ]


def test_calculate_correlations(cols):
    trn, hol, syn = cols
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    # prefix some columns with "tgt::"
    columns = [f"tgt::{c}" if c != "cat" else c for idx, c in enumerate(trn.columns)]
    trn.columns, syn.columns = columns, columns
    corr_trn = calculate_correlations(trn)
    exp_corr_trn = pd.DataFrame(
        [
            [1.0, 0.933898],
            [0.933898, 1.0],
        ],
        columns=["tgt::num", "tgt::dt"],
        index=["tgt::num", "tgt::dt"],
    )
    assert_frame_equal(corr_trn, exp_corr_trn, rtol=1e-4)
    corr_syn = calculate_correlations(syn, corr_trn.columns)
    exp_corr_syn = pd.DataFrame(
        [
            [1.0, 0.643165],
            [0.643165, 1.0],
        ],
        columns=["tgt::num", "tgt::dt"],
        index=["tgt::num", "tgt::dt"],
    )
    assert_frame_equal(corr_syn, exp_corr_syn, rtol=1e-4)


def test_plot_store_correlation_matrices(cols, workspace):
    trn, hol, syn = cols
    trn, bins = bin_data(trn, 3)
    syn, _ = bin_data(syn, bins)
    # prefix some columns with "tgt::"
    columns = [f"tgt::{c}" if idx > 0 else c for idx, c in enumerate(trn.columns)]
    trn.columns, syn.columns = columns, columns
    corr_trn = calculate_correlations(trn)
    corr_syn = calculate_correlations(trn, corr_trn.columns)
    plot_store_correlation_matrices(corr_trn, corr_syn, workspace)
    output_dir = workspace.workspace_dir / "figures"
    assert len(list(output_dir.glob("*.html"))) == 1
