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

import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_config

from mostlyai.qa._accuracy import (
    calculate_accuracy,
    calculate_accuracy_cnts,
    plot_univariate,
)
from mostlyai.qa._filesystem import TemporaryWorkspace


def calculate_distinct_categories_per_sequence(df: pd.DataFrame, context_key: str) -> pd.DataFrame:
    return df.groupby(context_key).nunique().reset_index(drop=True)


def calculate_distinct_categories_per_sequence_accuracy(
    ori_binned_cats_per_seq: pd.DataFrame, syn_binned_cats_per_seq: pd.DataFrame
) -> pd.DataFrame:
    acc_cats_per_seq = pd.DataFrame({"column": ori_binned_cats_per_seq.columns})
    with parallel_config("loky", n_jobs=min(16, max(1, cpu_count() - 1))):
        results = Parallel()(
            delayed(calculate_accuracy)(
                ori_bin_cols=ori_binned_cats_per_seq[[row["column"]]],
                syn_bin_cols=syn_binned_cats_per_seq[[row["column"]]],
            )
            for _, row in acc_cats_per_seq.iterrows()
        )
        acc_cats_per_seq["accuracy"], acc_cats_per_seq["accuracy_max"] = zip(*results)
    return acc_cats_per_seq


def plot_store_distinct_categories_per_sequence(
    ori_cats_per_seq_kdes: dict[str, pd.Series],
    syn_cats_per_seq_kdes: dict[str, pd.Series],
    ori_binned_cats_per_seq_cnts: dict[str, pd.Series],
    syn_binned_cats_per_seq_cnts: dict[str, pd.Series],
    acc_cats_per_seq: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    with parallel_config("loky", n_jobs=min(16, max(1, cpu_count() - 1))):
        Parallel()(
            delayed(plot_store_single_distinct_categories_per_sequence)(
                row["column"],
                ori_cats_per_seq_kdes.get(row["column"]),
                syn_cats_per_seq_kdes.get(row["column"]),
                ori_binned_cats_per_seq_cnts.get(row["column"]),
                syn_binned_cats_per_seq_cnts.get(row["column"]),
                row["accuracy"],
                workspace,
            )
            for _, row in acc_cats_per_seq.iterrows()
        )


def plot_store_single_distinct_categories_per_sequence(
    col: str,
    ori_cats_per_seq_kde: pd.Series,
    syn_cats_per_seq_kde: pd.Series,
    ori_binned_cats_per_seq_cnts: pd.Series,
    syn_binned_cats_per_seq_cnts: pd.Series,
    accuracy: float,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_univariate(
        col_name=col,
        ori_num_kde=ori_cats_per_seq_kde,
        syn_num_kde=syn_cats_per_seq_kde,
        ori_cat_col_cnts=None,
        syn_cat_col_cnts=None,
        ori_bin_col_cnts=ori_binned_cats_per_seq_cnts,
        syn_bin_col_cnts=syn_binned_cats_per_seq_cnts,
        accuracy=accuracy,
    )
    workspace.store_figure_html(fig, "distinct_categories_per_sequence", col)


def calculate_sequences_per_distinct_category(
    df: pd.DataFrame, context_key: str, top_cats: dict[str, list[str]] | None = None
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, list[str]], int]:
    seqs_per_cat = {
        col: df.groupby(col, observed=False)[context_key].nunique().rename_axis("index")
        for col in df.columns
        if col != context_key
    }

    # transform df to contain:
    # - top n_top_cats categories w.r.t. frequency of belonging to sequences
    # - other_cat for all other categories
    df = df.copy()
    n_top_cats = 9
    other_cat = "(other)"
    if top_cats is None:
        top_cats = {}
    for col in df.columns:
        if col == context_key:
            continue
        col_top_cats = top_cats.setdefault(col, seqs_per_cat[col].nlargest(n_top_cats).index.tolist())
        not_in_top_cats_mask = ~df[col].isin(col_top_cats)
        if not_in_top_cats_mask.any():
            if other_cat not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(other_cat)
            df.loc[not_in_top_cats_mask, col] = other_cat
            df[col] = df[col].cat.remove_unused_categories()
    binned_seqs_per_cat = {
        col: df.groupby(col, observed=False)[context_key].nunique().rename_axis("index")
        for col in df.columns
        if col != context_key
    }

    # number of sequences
    n_seqs = df[context_key].nunique()

    return seqs_per_cat, binned_seqs_per_cat, top_cats, n_seqs


def calculate_sequences_per_distinct_category_accuracy(
    *,
    ori_seqs_per_top_cat_cnts: dict[str, pd.Series],
    syn_seqs_per_top_cat_cnts: dict[str, pd.Series],
) -> pd.DataFrame:
    acc_seq_per_cat = pd.DataFrame({"column": ori_seqs_per_top_cat_cnts.keys()})
    with parallel_config("loky", n_jobs=min(16, max(1, cpu_count() - 1))):
        results = Parallel()(
            delayed(calculate_accuracy_cnts)(
                ori_seqs_per_top_cat_cnts[row["column"]],
                syn_seqs_per_top_cat_cnts[row["column"]],
            )
            for _, row in acc_seq_per_cat.iterrows()
        )
        acc_seq_per_cat["accuracy"], acc_seq_per_cat["accuracy_max"] = zip(*results)
    return acc_seq_per_cat


def plot_store_sequences_per_distinct_category(
    ori_seqs_per_cat_cnts: dict[str, pd.Series],
    syn_seqs_per_cat_cnts: dict[str, pd.Series],
    ori_seqs_per_top_cat_cnts: dict[str, pd.Series],
    syn_seqs_per_top_cat_cnts: dict[str, pd.Series],
    ori_n_seqs: int,
    syn_n_seqs: int,
    acc_seqs_per_cat: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    with parallel_config("loky", n_jobs=min(16, max(1, cpu_count() - 1))):
        Parallel()(
            delayed(plot_store_single_sequences_per_distinct_category)(
                row["column"],
                ori_seqs_per_cat_cnts.get(row["column"]),
                syn_seqs_per_cat_cnts.get(row["column"]),
                ori_seqs_per_top_cat_cnts.get(row["column"]),
                syn_seqs_per_top_cat_cnts.get(row["column"]),
                ori_n_seqs,
                syn_n_seqs,
                row["accuracy"],
                workspace,
            )
            for _, row in acc_seqs_per_cat.iterrows()
        )


def plot_store_single_sequences_per_distinct_category(
    col: str,
    ori_seqs_per_cat_cnts: pd.Series,
    syn_seqs_per_cat_cnts: pd.Series,
    ori_seqs_per_top_cat_cnts: pd.Series,
    syn_seqs_per_top_cat_cnts: pd.Series,
    ori_n_seqs: int,
    syn_n_seqs: int,
    accuracy: float,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_univariate(
        col_name=col,
        ori_num_kde=None,
        syn_num_kde=None,
        ori_cat_col_cnts=ori_seqs_per_cat_cnts,
        syn_cat_col_cnts=syn_seqs_per_cat_cnts,
        ori_bin_col_cnts=ori_seqs_per_top_cat_cnts,
        syn_bin_col_cnts=syn_seqs_per_top_cat_cnts,
        ori_cnt=ori_n_seqs,
        syn_cnt=syn_n_seqs,
        accuracy=accuracy,
        sort_categorical_binned_by_frequency=False,
        max_label_length=15,
    )
    workspace.store_figure_html(fig, "sequences_per_distinct_category", col)
