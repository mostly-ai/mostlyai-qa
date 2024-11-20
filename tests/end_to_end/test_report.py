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

import os
import uuid
from pathlib import Path

import pandas as pd

from mostlyai.qa.report_from_statistics import report_from_statistics
from mostlyai.qa.report import report

baseball_path = Path(os.path.realpath(__file__)).parent / "fixtures" / "baseball"
census_path = Path(os.path.realpath(__file__)).parent / "fixtures" / "census"


def test_report_flat(tmp_path):
    statistics_path = tmp_path / "statistics"
    syn_tgt_data = pd.read_parquet(census_path / "census-synthetic.parquet")
    trn_tgt_data = pd.read_parquet(census_path / "census-training.parquet")
    hol_tgt_data = pd.read_parquet(census_path / "census-holdout.parquet")
    report_path, metrics = report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        statistics_path=statistics_path,
        max_sample_size_accuracy=2000,
        max_sample_size_embeddings=200,
    )

    assert report_path.exists()

    accuracy = metrics["accuracy"]
    assert 0.5 <= accuracy["overall"] <= 1.0
    assert 0.5 <= accuracy["univariate"] <= 1.0
    assert 0.5 <= accuracy["bivariate"] <= 1.0
    assert accuracy["coherence"] is None
    assert 0.8 <= accuracy["overall_max"] <= 1.0
    assert 0.8 <= accuracy["univariate_max"] <= 1.0
    assert 0.8 <= accuracy["bivariate_max"] <= 1.0

    similarity = metrics["similarity"]
    assert 0.8 <= similarity["cosine_similarity_training_synthetic"] <= 1.0
    assert 0.0 <= similarity["discriminator_auc_training_synthetic"] <= 1.0
    assert 0.8 <= similarity["cosine_similarity_training_holdout"] <= 1.0
    assert 0.0 <= similarity["discriminator_auc_training_holdout"] <= 1.0

    distances = metrics["distances"]
    assert 0 <= distances["ims_training"] <= 1.0
    assert 0 <= distances["dcr_training"] <= 1.0
    assert 0 <= distances["ims_holdout"] <= 1.0
    assert 0 <= distances["dcr_holdout"] <= 1.0
    assert 0 <= distances["dcr_share"] <= 1.0

    report_path = report_from_statistics(
        syn_tgt_data=syn_tgt_data,
        statistics_path=statistics_path,
        max_sample_size_accuracy=2000,
        max_sample_size_embeddings=200,
    )

    assert report_path.exists()


def test_report_sequential(tmp_path):
    statistics_path = tmp_path / "statistics"
    report_path = Path(tmp_path / "my-report.html")

    trn_tgt_data = pd.read_parquet(baseball_path / "seasons-training.parquet")
    hol_tgt_data = pd.read_parquet(baseball_path / "seasons-holdout.parquet")
    syn_tgt_data = pd.read_parquet(baseball_path / "seasons-synthetic.parquet")
    trn_ctx_data = pd.read_parquet(baseball_path / "players-training.parquet")
    hol_ctx_data = pd.read_parquet(baseball_path / "players-holdout.parquet")
    syn_ctx_data = pd.concat([trn_ctx_data, hol_ctx_data])

    report_path, metrics = report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        syn_ctx_data=syn_ctx_data,
        trn_ctx_data=trn_ctx_data,
        hol_ctx_data=hol_ctx_data,
        ctx_primary_key="id",
        tgt_context_key="players_id",
        report_path=report_path,
        statistics_path=statistics_path,
        max_sample_size_accuracy=2000,
        max_sample_size_embeddings=200,
    )

    assert report_path.exists()

    accuracy = metrics["accuracy"]
    assert 0.8 <= accuracy["overall"] <= 1.0
    assert 0.8 <= accuracy["univariate"] <= 1.0
    assert 0.8 <= accuracy["bivariate"] <= 1.0
    assert 0.8 <= accuracy["coherence"] <= 1.0
    assert 0.8 <= accuracy["overall_max"] <= 1.0
    assert 0.8 <= accuracy["univariate_max"] <= 1.0
    assert 0.8 <= accuracy["bivariate_max"] <= 1.0
    assert 0.8 <= accuracy["coherence_max"] <= 1.0

    similarity = metrics["similarity"]
    assert 0.8 <= similarity["cosine_similarity_training_synthetic"] <= 1.0
    assert 0.0 <= similarity["discriminator_auc_training_synthetic"] <= 1.0
    assert 0.8 <= similarity["cosine_similarity_training_holdout"] <= 1.0
    assert 0.0 <= similarity["discriminator_auc_training_holdout"] <= 1.0

    distances = metrics["distances"]
    assert 0 <= distances["ims_training"] <= 1.0
    assert 0 <= distances["dcr_training"] <= 1.0
    assert 0 <= distances["ims_holdout"] <= 1.0
    assert 0 <= distances["dcr_holdout"] <= 1.0
    assert 0 <= distances["dcr_share"] <= 1.0

    report_path = report_from_statistics(
        syn_tgt_data=syn_tgt_data,
        syn_ctx_data=syn_ctx_data,
        ctx_primary_key="id",
        tgt_context_key="players_id",
        max_sample_size_accuracy=3000,
        max_sample_size_embeddings=300,
        statistics_path=statistics_path,
    )

    assert report_path.exists()


def test_report_flat_rare(tmp_path):
    statistics_path = Path(tmp_path / "statistics")

    # test case where all values are rare category protected
    syn_tgt_data = pd.DataFrame({"x": ["_RARE_" for _ in range(100)]})
    trn_tgt_data = pd.DataFrame({"x": [str(uuid.uuid4()) for _ in range(100)]})
    hol_tgt_data = pd.DataFrame({"x": [str(uuid.uuid4()) for _ in range(100)]})
    _, metrics = report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        statistics_path=statistics_path,
    )
    assert metrics["accuracy"]["univariate"] == 0.0
    assert metrics["distances"]["ims_training"] == metrics["distances"]["ims_holdout"] == 0.0

    # test case where rare values are not protected, and we leak trn into synthetic
    syn_tgt_data = pd.DataFrame({"x": trn_tgt_data["x"].sample(100, replace=True)})
    _, metrics = report(
        syn_tgt_data=syn_tgt_data,
        trn_tgt_data=trn_tgt_data,
        hol_tgt_data=hol_tgt_data,
        statistics_path=statistics_path,
    )
    assert metrics["distances"]["ims_training"] > metrics["distances"]["ims_holdout"]
    assert metrics["distances"]["dcr_training"] < metrics["distances"]["dcr_holdout"]


def test_report_flat_early_exit(tmp_path):
    # test early exit for dfs with <100 rows
    df = pd.DataFrame({"col": list(range(99))})
    _, metrics = report(syn_tgt_data=df, trn_tgt_data=df, hol_tgt_data=df)
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
        # other setups should produce report
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100), "early_term": False},
        {"dfs": make_dfs(ctx_rows=100, tgt_rows=100, ctx_cols=["ctx_col"], tgt_cols=["tgt_col"]), "early_term": False},
    ]

    for test_idx, df_dict in enumerate(test_dfs):
        ctx_df, tgt_df = df_dict.pop("dfs")
        syn_ctx_data = trn_ctx_data = val_ctx_data = ctx_df
        syn_tgt_data = trn_tgt_data = val_tgt_data = tgt_df
        early_term = df_dict.pop("early_term")
        _, metrics = report(
            syn_tgt_data=syn_tgt_data,
            trn_tgt_data=trn_tgt_data,
            hol_tgt_data=val_tgt_data,
            syn_ctx_data=syn_ctx_data,
            trn_ctx_data=trn_ctx_data,
            hol_ctx_data=val_ctx_data,
            tgt_context_key="ck",
            ctx_primary_key="pk",
        )
        assert metrics is None if early_term else metrics is not None, f"Test {test_idx} failed"


def test_report_sequential_few_records(tmp_path):
    # ensure that we don't crash in case of dominant zero-seq-length
    ctx = pd.DataFrame({"id": list(range(1000))})
    tgt = pd.DataFrame({"id": [1, 2, 3, 4, 5] * 100, "col": ["a"] * 500})
    _, metrics = report(
        syn_tgt_data=tgt,
        trn_tgt_data=tgt,
        hol_tgt_data=tgt,
        syn_ctx_data=ctx,
        trn_ctx_data=ctx,
        hol_ctx_data=ctx,
        tgt_context_key="id",
        ctx_primary_key="id",
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
    path, metrics = report(
        syn_tgt_data=df,
        trn_tgt_data=df,
        statistics_path=tmp_path / "stats",
    )
    assert metrics is not None
    path = report_from_statistics(
        syn_tgt_data=df,
        statistics_path=tmp_path / "stats",
    )
    assert path is not None
