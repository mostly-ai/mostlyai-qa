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

import uuid
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from mostlyai import qa
from datetime import datetime, timedelta


def mock_data(n):
    df = pd.DataFrame(
        {
            "int": pd.Series(np.random.choice(list(range(10)) + [np.nan], n), dtype="Int64[pyarrow]"),
            "float": pd.Series(
                np.random.choice(list(np.random.uniform(size=10)) + [np.nan], n), dtype="float64[pyarrow]"
            ),
            "cat": pd.Series(np.random.choice(["f", "m", np.nan], n), dtype="string[pyarrow]"),
            "bool": pd.Series(np.random.choice([True, False, np.nan], n), dtype="bool[pyarrow]"),
            "date": pd.Series(np.random.choice([np.datetime64("today", "D"), np.nan], n), dtype="datetime64[ns]"),
            "text": pd.Series([str(uuid.uuid4())[:4] for _ in range(n)], dtype="object"),
        }
    )
    return df


def test_report_flat(tmp_path):
    statistics_path = tmp_path / "statistics"
    syn_tgt_data = mock_data(220)
    trn_tgt_data = mock_data(180)
    hol_tgt_data = mock_data(140)
    report_path, metrics = qa.report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        statistics_path=statistics_path,
        max_sample_size_accuracy=120,
        max_sample_size_embeddings=80,
    )

    assert report_path.exists()

    accuracy = metrics.accuracy
    assert 0.3 <= accuracy.overall <= 1.0
    assert 0.3 <= accuracy.univariate <= 1.0
    assert 0.3 <= accuracy.bivariate <= 1.0
    assert accuracy.coherence is None
    assert 0.3 <= accuracy.overall_max <= 1.0
    assert 0.3 <= accuracy.univariate_max <= 1.0
    assert 0.3 <= accuracy.bivariate_max <= 1.0

    similarity = metrics.similarity
    assert 0.8 <= similarity.cosine_similarity_training_synthetic <= 1.0
    assert 0.0 <= similarity.discriminator_auc_training_synthetic <= 1.0
    assert 0.8 <= similarity.cosine_similarity_training_holdout <= 1.0
    assert 0.0 <= similarity.discriminator_auc_training_holdout <= 1.0

    distances = metrics.distances
    assert 0 <= distances.ims_training <= 1.0
    assert 0 <= distances.dcr_training <= 1.0
    assert 0 <= distances.ims_holdout <= 1.0
    assert 0 <= distances.dcr_holdout <= 1.0
    assert 0 <= distances.dcr_share <= 1.0

    report_path = qa.report_from_statistics(
        syn_tgt_data=syn_tgt_data,
        statistics_path=statistics_path,
        max_sample_size_accuracy=110,
        max_sample_size_embeddings=70,
    )

    assert report_path.exists()


def test_report_sequential(tmp_path):
    statistics_path = tmp_path / "statistics"
    report_path = Path(tmp_path / "my-report.html")

    # generate mock context data
    syn_ctx_data = mock_data(220).reset_index(names="id")
    trn_ctx_data = mock_data(180).reset_index(names="id")
    hol_ctx_data = mock_data(140).reset_index(names="id")

    # generate mock sequential target data
    syn_tgt_data = mock_data(220 * 3)
    syn_tgt_data["ctx_id"] = np.random.choice(syn_ctx_data["id"], 220 * 3)
    trn_tgt_data = mock_data(180 * 4)
    trn_tgt_data["ctx_id"] = np.random.choice(trn_ctx_data["id"], 180 * 4)
    hol_tgt_data = mock_data(140 * 4)
    hol_tgt_data["ctx_id"] = np.random.choice(hol_ctx_data["id"], 140 * 4)

    # generate report
    report_path, metrics = qa.report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        syn_ctx_data=syn_ctx_data,
        trn_ctx_data=trn_ctx_data,
        hol_ctx_data=hol_ctx_data,
        ctx_primary_key="id",
        tgt_context_key="ctx_id",
        report_path=report_path,
        statistics_path=statistics_path,
        max_sample_size_accuracy=120,
        max_sample_size_embeddings=80,
        max_sample_size_coherence=30,
    )

    assert report_path.exists()

    accuracy = metrics.accuracy
    assert 0.3 <= accuracy.overall <= 1.0
    assert 0.3 <= accuracy.univariate <= 1.0
    assert 0.3 <= accuracy.bivariate <= 1.0
    assert 0.3 <= accuracy.coherence <= 1.0
    assert 0.3 <= accuracy.overall_max <= 1.0
    assert 0.3 <= accuracy.univariate_max <= 1.0
    assert 0.3 <= accuracy.bivariate_max <= 1.0
    assert 0.3 <= accuracy.coherence_max <= 1.0

    similarity = metrics.similarity
    assert 0.3 <= similarity.cosine_similarity_training_synthetic <= 1.0
    assert 0.0 <= similarity.discriminator_auc_training_synthetic <= 1.0
    assert 0.3 <= similarity.cosine_similarity_training_holdout <= 1.0
    assert 0.0 <= similarity.discriminator_auc_training_holdout <= 1.0

    distances = metrics.distances
    assert 0 <= distances.ims_training <= 1.0
    assert 0 <= distances.dcr_training <= 1.0
    assert 0 <= distances.ims_holdout <= 1.0
    assert 0 <= distances.dcr_holdout <= 1.0
    assert 0 <= distances.dcr_share <= 1.0

    report_path = qa.report_from_statistics(
        syn_tgt_data=syn_tgt_data,
        syn_ctx_data=syn_ctx_data,
        ctx_primary_key="id",
        tgt_context_key="ctx_id",
        max_sample_size_accuracy=130,
        max_sample_size_embeddings=90,
        max_sample_size_coherence=30,
        statistics_path=statistics_path,
    )

    assert report_path.exists()


def test_report_flat_rare(tmp_path):
    statistics_path = Path(tmp_path / "statistics")

    # test case where all values are rare category protected
    syn_tgt_data = pd.DataFrame({"x": ["_RARE_" for _ in range(100)]})
    trn_tgt_data = pd.DataFrame({"x": [str(uuid.uuid4()) for _ in range(100)]})
    hol_tgt_data = pd.DataFrame({"x": [str(uuid.uuid4()) for _ in range(100)]})
    _, metrics = qa.report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        statistics_path=statistics_path,
    )
    assert metrics.accuracy.univariate == 0.0
    assert metrics.distances.ims_training == metrics.distances.ims_holdout == 0.0


def test_report_flat_early_exit(tmp_path):
    # test early exit for dfs with <100 rows
    df = pd.DataFrame({"col": list(range(99))})
    _, metrics = qa.report(syn_tgt_data=df, trn_tgt_data=df, hol_tgt_data=df)
    assert metrics is None


def test_report_sequential_early_exit(tmp_path):
    def make_dfs(
        ctx_rows: int, tgt_rows: int, ctx_cols: list[str] = None, tgt_cols: list[str] = None, shift: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ctx_cols = ctx_cols or []
        tgt_cols = tgt_cols or []
        ctx = pd.DataFrame({"pk": range(ctx_rows)} | {c: range(ctx_rows) for c in ctx_cols})
        tgt = pd.DataFrame({"ck": list(range(shift, shift + tgt_rows))} | {c: range(tgt_rows) for c in tgt_cols})
        return ctx, tgt

    # test empty-ish data sets
    test_dfs = [
        # setups with <100 rows in tgt/ctx should early terminate
        {"dfs": make_dfs(ctx_rows=99, tgt_rows=99, ctx_cols=["ctx_col"], tgt_cols=["tgt_col"]), "early_term": True},
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100, shift=90, tgt_cols=["tgt_col"]), "early_term": True},
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100, shift=100, tgt_cols=["tgt_col"]), "early_term": True},
        # other setups should produce report
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100), "early_term": False},
        {"dfs": make_dfs(ctx_rows=101, tgt_rows=100), "early_term": False},
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100, ctx_cols=["ctx_col"], tgt_cols=["tgt_col"]), "early_term": False},
    ]

    for test_idx, df_dict in enumerate(test_dfs):
        ctx_df, tgt_df = df_dict.pop("dfs")
        syn_ctx_data = trn_ctx_data = hol_ctx_data = ctx_df
        syn_tgt_data = trn_tgt_data = hol_tgt_data = tgt_df
        early_term = df_dict.pop("early_term")
        _, metrics = qa.report(
            syn_tgt_data=syn_tgt_data,
            trn_tgt_data=trn_tgt_data,
            hol_tgt_data=hol_tgt_data,
            syn_ctx_data=syn_ctx_data,
            trn_ctx_data=trn_ctx_data,
            hol_ctx_data=hol_ctx_data,
            tgt_context_key="ck",
            ctx_primary_key="pk",
        )
        assert metrics is None if early_term else metrics is not None, f"Test {test_idx} failed"


def test_report_few_holdout_records(tmp_path):
    tgt = pd.DataFrame({"id": list(range(100)), "col": ["a"] * 100})
    _, metrics = qa.report(
        syn_tgt_data=tgt,
        trn_tgt_data=tgt,
        hol_tgt_data=tgt[:10],
    )
    assert metrics is not None


def test_odd_column_names(tmp_path):
    values = ["a", "b"] * 50
    df = pd.DataFrame(
        {
            "some.test": values,
            "foo%bar|this-long{c[u]rly} *": values,
            "3": values,
        }
    )
    path, metrics = qa.report(
        syn_tgt_data=df,
        trn_tgt_data=df,
        statistics_path=tmp_path / "stats",
    )
    assert metrics is not None
    path = qa.report_from_statistics(
        syn_tgt_data=df,
        statistics_path=tmp_path / "stats",
    )
    assert path is not None


def test_missing(tmp_path):
    df1 = mock_data(100)
    df2 = df1.copy()
    df2.loc[:, :] = np.nan
    _, metrics = qa.report(
        syn_tgt_data=df1,
        trn_tgt_data=df2,
    )
    assert metrics is not None
    _, metrics = qa.report(
        syn_tgt_data=df2,
        trn_tgt_data=df1,
    )
    assert metrics is not None


def test_mixed_dtypes(tmp_path):
    # test that datetime columns drawn from the same distribution, but having different dtype
    # are still yielding somewhat good results and warning is issued

    def generate_dates(start_date, end_date, num_samples):
        days_range = (end_date - start_date).days
        return [start_date + timedelta(days=int(days)) for days in np.random.randint(0, days_range, num_samples)]

    num_samples = 200
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    df = pd.DataFrame(
        {
            "trn_dt": pd.Series(generate_dates(start_date, end_date, num_samples)).values.astype(str),
            "syn_dt": pd.Series(generate_dates(start_date, end_date, num_samples), dtype="datetime64[ns]"),
        }
    )
    trn_df, syn_df = df["trn_dt"].to_frame("dt"), df["syn_dt"].to_frame("dt")

    with warnings.catch_warnings(record=True) as w:
        _, statistics = qa.report(
            syn_tgt_data=syn_df,
            trn_tgt_data=trn_df,
            report_path=tmp_path / "report.html",
        )
        expected_warning = (
            "The column(s) ['dt'] have inconsistent data types across `syn`, `trn`, and `hol`. "
            "To achieve the most accurate results, please harmonize the data types of these inputs. "
            "Proceeding with a best-effort attempt..."
        )
        assert any(expected_warning in str(warning.message) for warning in w), (
            "Expected a warning about dtype mismatch for column 'dt'"
        )
    assert statistics.accuracy.overall > 0.6
    assert 0.4 < statistics.similarity.discriminator_auc_training_synthetic < 0.6
