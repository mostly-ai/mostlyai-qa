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

import functools
import hashlib
import logging
import math
from typing import Any, Callable, Iterable

import fastcluster
import numpy as np
import pandas as pd
import phik.phik
import scipy.cluster
import scipy.stats
from joblib import Parallel, delayed, parallel_config, cpu_count

from mostlyai.qa.common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
    EMPTY_BIN,
    NA_BIN,
    MIN_RARE_CAT_PROTECTION,
    OTHER_BIN,
    RARE_BIN,
    MAX_UNIVARIATE_PLOTS,
    MAX_BIVARIATE_TGT_PLOTS,
    MAX_BIVARIATE_CTX_PLOTS,
    MAX_BIVARIATE_NXT_PLOTS,
    CTX_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
    NXT_COLUMN_PREFIX,
    MAX_ENGINE_RARE_CATEGORY_THRESHOLD,
    TGT_COLUMN,
)
from plotly import graph_objs as go

from mostlyai.qa.filesystem import TemporaryWorkspace, Statistics

_LOG = logging.getLogger(__name__)


def calculate_univariates(
    trn_bin: pd.DataFrame,
    syn_bin: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates univariate accuracies for all target columns.
    """

    tgt_cols = [c for c in trn_bin.columns if c.startswith(TGT_COLUMN)]
    accuracies = pd.DataFrame({"column": tgt_cols})
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        results = Parallel()(
            delayed(calculate_accuracy)(
                trn_bin_cols=trn_bin[[row["column"]]],
                syn_bin_cols=syn_bin[[row["column"]]],
            )
            for _, row in accuracies.iterrows()
        )
        accuracies["accuracy"], accuracies["accuracy_max"] = zip(*results)
    return accuracies


def calculate_bivariates(
    trn_bin: pd.DataFrame,
    syn_bin: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates bivariate accuracies.
    Each target column is paired with:
     - another target column (tgt:col1, tgt:col2),
     - each context column (tgt:col1, ctx:col), and
     - corresponding next column (tgt:col1, nxt:col1).
    For each such column pair, value pair frequencies
    are calculated both for training and synthetic data.
    """

    # the result for symmetric pairs is the same, so we only calculate one of them
    # later, we append copy results for symmetric pairs
    accuracies = calculate_bivariate_columns(trn_bin, append_symetric=False)

    # calculate bivariates if there is at least one pair
    if len(accuracies) > 0:
        with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
            results = Parallel()(
                delayed(calculate_accuracy)(
                    trn_bin_cols=trn_bin[[row["col1"], row["col2"]]],
                    syn_bin_cols=syn_bin[[row["col1"], row["col2"]]],
                )
                for _, row in accuracies.iterrows()
            )
        accuracies["accuracy"], accuracies["accuracy_max"] = zip(*results)
    else:
        # enforce consistent columns
        accuracies[["accuracy", "accuracy_max"]] = None
        # ensure required number of progress messages are sent

    accuracies = pd.concat(
        [
            accuracies,
            accuracies.rename(columns={"col2": "col1", "col1": "col2"}),
        ],
        axis=0,
    ).reset_index(drop=True)

    return accuracies


def calculate_bivariate_columns(trn_bin: pd.DataFrame, append_symetric: bool = True) -> pd.DataFrame:
    """
    Creates DataFrame with all column-pairs subject to bivariate analysis.
    """

    tgt_cols = [c for c in trn_bin.columns if c.startswith(TGT_COLUMN_PREFIX)]
    ctx_cols = [c for c in trn_bin.columns if c.startswith(CTX_COLUMN_PREFIX)]
    nxt_cols = [c for c in trn_bin.columns if c.startswith(NXT_COLUMN_PREFIX)]

    # create cross-combinations between all `tgt` columns
    columns_df = pd.merge(
        pd.DataFrame({"col1": tgt_cols}),
        pd.DataFrame({"col2": tgt_cols}),
        how="cross",
    ).assign(type="tgt")
    columns_df = columns_df.loc[columns_df.col1 < columns_df.col2]

    # create combinations between all `tgt` and all `ctx` columns
    if len(ctx_cols) > 0:
        ctx_accuracies = pd.merge(
            pd.DataFrame({"col1": tgt_cols}),
            pd.DataFrame({"col2": ctx_cols}),
            how="cross",
        ).assign(type="ctx")
        columns_df = pd.concat([columns_df, ctx_accuracies], axis=0).reset_index(drop=True)

    # create combinations between all `tgt` and their corresponding `ntx` column
    if len(nxt_cols) > 0:
        nxt_accuracies = pd.merge(
            pd.DataFrame({"col1": tgt_cols}),
            pd.DataFrame({"col2": nxt_cols}),
            how="cross",
        ).assign(type="nxt")
        nxt_accuracies = nxt_accuracies.loc[nxt_accuracies.col1.str[4:] == nxt_accuracies.col2.str[4:]]
        columns_df = pd.concat([columns_df, nxt_accuracies], axis=0).reset_index(drop=True)

    if append_symetric:
        # calculate symmetric combinations
        columns_df = pd.concat(
            [
                columns_df,
                columns_df.rename(columns={"col2": "col1", "col1": "col2"}),
            ],
            axis=0,
        ).reset_index(drop=True)

    return columns_df


def calculate_expected_l1_multinomial(probs: list[float], n_1: int, n_2: int) -> np.float64:
    """
    Calculate expected L1 distance for two multinomial samples of size `n_1` and `n_2` that follow `probs`.
    """

    def calculate_expected_l1_binomial(p: float, n_1: int, n_2: int):
        # the variance of a binomial is p*(1-p)/n
        variance_1 = p * (1 - p) / n_1
        variance_2 = p * (1 - p) / n_2
        # the difference between two normal distributions is normally distributed, and has a variance that is the sum
        # of the variances; see https://stats.stackexchange.com/a/186545
        variance = variance_1 + variance_2
        # the expected mean of the absolute value of a normal distribution is sqrt(2/PI);
        # https://en.wikipedia.org/wiki/Half-normal_distribution
        expected_l1 = np.sqrt(variance * 2 / np.pi)
        return expected_l1

    # we sum expectations across each prob
    expected_l1 = np.sum([calculate_expected_l1_binomial(p, n_1, n_2) for p in probs])
    return expected_l1


def calculate_accuracy(trn_bin_cols: pd.DataFrame, syn_bin_cols: pd.DataFrame) -> tuple[np.float64, np.float64]:
    """
    Calculates accuracy between the empirical distributions of training vs. synthetic, as well as the max accuracy,
    that can be expected due to the sampling noise.
    """

    # create relative frequency tables for `trn` and `syn`
    trn_freq = trn_bin_cols.value_counts(normalize=True)
    syn_freq = syn_bin_cols.value_counts(normalize=True)
    freq = pd.merge(
        trn_freq.to_frame("tgt").reset_index(),
        syn_freq.to_frame("syn").reset_index(),
        how="outer",
        on=list(trn_bin_cols.columns),
    )
    freq["tgt"] = freq["tgt"].fillna(0.0)
    freq["syn"] = freq["syn"].fillna(0.0)
    # calculate L1 distance between `trn` and `syn`
    observed_l1 = (freq["tgt"] - freq["syn"]).abs().sum()
    # calculated expected L1 distance based on `trn`
    n_trn = trn_bin_cols.shape[0]
    n_syn = syn_bin_cols.shape[0]
    expected_l1 = calculate_expected_l1_multinomial(freq["tgt"].to_list(), n_trn, n_syn)
    # convert to accuracy; trim superfluous precision
    observed_acc = (1 - observed_l1 / 2).round(5)
    expected_acc = (1 - expected_l1 / 2).round(5)
    return observed_acc, expected_acc


def calculate_numeric_uni_kdes(df: pd.DataFrame, trn_kdes: dict[str, pd.Series] | None = None) -> dict[str, pd.Series]:
    """
    Calculates univariate kernel density estimates for numeric/datetime columns.
    `trn_kdes` is used as a reference for the grid points to evaluate the KDEs.
    """

    col_kdes = {}

    # calculate numeric/datetime column KDEs
    for col, series in df.items():
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_dtype(series)
        is_empty = series.dropna().size == 0
        missing_in_trn = trn_kdes is not None and col not in trn_kdes
        if not missing_in_trn and (is_numeric or is_datetime) and not is_empty:
            # column is treated as numeric/datetime

            if trn_kdes is None:
                # determine grid points to evaluate kernel estimates
                val_min = series.min()
                val_max = series.max()
                no_of_bins = 100
                if is_datetime:
                    val_x = pd.date_range(start=val_min, end=val_max, periods=no_of_bins + 1)
                else:
                    val_x = pd.Series(np.linspace(val_min, val_max, no_of_bins), dtype="float")
            else:  # trn_kdes is not None
                # use grid points from training data
                val_x = trn_kdes[col].index

            # estimate gaussian kernels
            series_vals = series.dropna().to_numpy("float")
            if len(series_vals) > 1:
                try:
                    series_kde = scipy.stats.gaussian_kde(series_vals)
                    val_y = series_kde(val_x.to_numpy("float"))
                    val_y = (val_y / (val_y.sum() + 1e-30)).round(5)
                except np.linalg.LinAlgError:
                    # handle `singular matrix` error that can occur for constants
                    val_y = [1] * len(val_x)
            elif len(series_vals) == 1:
                val_y = [1] * len(val_x)
            else:
                val_y = [np.nan] * len(val_x)
            col_kdes[col] = pd.Series(val_y, index=val_x, name=col)

    if trn_kdes is not None:
        # ensure the result has the same shape as for Model QA
        for trn_col, trn_kdes in trn_kdes.items():
            if trn_col not in col_kdes:
                kdes = pd.Series(0, index=trn_kdes.index, name=trn_col)
                col_kdes[trn_col] = kdes

    return col_kdes


def calculate_categorical_uni_counts(
    df: pd.DataFrame,
    trn_col_counts: dict[str, pd.Series] | None = None,
    hash_rare_values: bool = True,
) -> dict[str, pd.Series]:
    """
    Calculates counts of unique values in each categorical column of a DataFrame.
    Protects rare labels by hashing them. `trn_col_counts` is used as
    template to ensure the result has the same shape as for Model QA.
    """

    def hash_rare(rare: Any, column_name: str) -> str:
        hash_in = str(rare) + column_name
        hash_obj = hashlib.md5()
        hash_obj.update(hash_in.encode("utf-8"))
        hash_out = hash_obj.hexdigest()
        protected = "rare_" + hash_out
        return protected

    col_counts = {}

    # calculate categorical column counts
    for col, series in df.items():
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_dtype(series)
        is_empty = series.dropna().size == 0
        if is_empty or not (is_numeric or is_datetime):
            # column is treated as categorical
            if is_empty:
                # prerequsite for hashing is to deal with string-typed columns
                series = series.astype("string")
            cnts = series.value_counts(sort=False, dropna=False).to_frame("cnt").reset_index()
            if hash_rare_values:
                # mask rare categories (for privacy reasons)
                # using hashing enables us to map the same rare categories to the same hash
                # across different runs
                # appending column name to the input of hashing function ensures hashes
                # are not repeated across columns
                cnts[col] = cnts[col].mask(
                    cnts["cnt"] <= MAX_ENGINE_RARE_CATEGORY_THRESHOLD,
                    (cnts[col]).apply(functools.partial(hash_rare, column_name=col)),
                )
            cnts = cnts.set_index(cnts[col].rename(None))["cnt"].rename(col)
            col_counts[col] = cnts

    if trn_col_counts is not None:
        # ensure the result has the same shape as for Model QA
        for trn_col, trn_counts in trn_col_counts.items():
            col_counts[trn_col] = col_counts.get(trn_col, pd.Series(0, index=trn_counts.index, name=trn_col))

    return col_counts


def bin_count_uni(col: str, values: pd.Series) -> tuple[str, pd.Series]:
    # remove unused categories (potentially (n/a) etc.), keep order of categories
    return col, values.cat.remove_unused_categories().value_counts(sort=False)


def bin_count_biv(col1: str, col2: str, x: pd.Series, y: pd.Series) -> tuple[tuple[str, str], pd.Series]:
    # keep order of categories
    return (col1, col2), pd.concat([x, y], axis=1).value_counts(sort=False)


def calculate_bin_counts(
    binned: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[tuple[str, str], pd.Series]]:
    """
    Calculates counts of unique values in each bin.
    """

    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        results = Parallel()(
            delayed(bin_count_uni)(
                col=col,
                values=values,
            )
            for col, values in binned.items()
        )
        bin_cnts_uni = dict(results)

    biv_cols = calculate_bivariate_columns(binned)
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        results = Parallel()(
            delayed(bin_count_biv)(
                col1=row["col1"],
                col2=row["col2"],
                x=binned[row["col1"]],
                y=binned[row["col2"]],
            )
            for _, row in biv_cols.iterrows()
        )
        bin_cnts_biv = dict(results)

    return bin_cnts_uni, bin_cnts_biv


def plot_store_univariates(
    trn_num_kdes: dict[str, pd.Series],
    syn_num_kdes: dict[str, pd.Series],
    trn_cat_cnts: dict[str, pd.Series],
    syn_cat_cnts: dict[str, pd.Series],
    trn_cnts_uni: dict[str, pd.Series],
    syn_cnts_uni: dict[str, pd.Series],
    acc_uni: pd.DataFrame,
    workspace: TemporaryWorkspace,
    show_accuracy: bool,
) -> None:
    """
    Plots all univariate accuracy figures and stores them under workspace dir.
    """

    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        Parallel()(
            delayed(plot_store_univariate)(
                row["column"],
                trn_num_kdes.get(row["column"]),
                syn_num_kdes.get(row["column"]),
                trn_cat_cnts.get(row["column"]),
                syn_cat_cnts.get(row["column"]),
                trn_cnts_uni[row["column"]],
                syn_cnts_uni[row["column"]],
                row["accuracy"] if show_accuracy else None,
                workspace,
            )
            for _, row in acc_uni.iterrows()
        )


def plot_store_univariate(
    col: str,
    trn_num_kde: pd.Series | None,
    syn_num_kde: pd.Series | None,
    trn_cat_col_cnts: pd.Series | None,
    syn_cat_col_cnts: pd.Series | None,
    trn_bin_col_cnts: pd.Series,
    syn_bin_col_cnts: pd.Series,
    accuracy: float | None,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_univariate(
        col,
        trn_num_kde,
        syn_num_kde,
        trn_cat_col_cnts,
        syn_cat_col_cnts,
        trn_bin_col_cnts,
        syn_bin_col_cnts,
        accuracy,
    )
    workspace.store_figure_html(fig, "univariate", col)


def plot_univariate(
    col_name: str,
    trn_num_kde: pd.Series | None,
    syn_num_kde: pd.Series | None,
    trn_cat_col_cnts: pd.Series | None,
    syn_cat_col_cnts: pd.Series | None,
    trn_bin_col_cnts: pd.Series,
    syn_bin_col_cnts: pd.Series,
    accuracy: float | None,
) -> go.Figure:
    # either numerical/datetime KDEs or categorical counts must be provided

    # plot title
    col_name = trim_label(col_name, max_length=30)
    plot_title = f"<b>{col_name}</b>" + (f" <sup>{accuracy:.1%}</sup>" if accuracy is not None else "")
    # plot layout
    layout = go.Layout(
        title=dict(text=plot_title, x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=220,
        margin=dict(l=10, r=10, b=10, t=40, pad=5),
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(
            showticklabels=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#999999",
            rangemode="tozero",
        ),
        yaxis2=dict(
            gridwidth=1,
            gridcolor="#d3d3d3",
            griddash="dot",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#999999",
            rangemode="tozero",
        ),
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=("distribution", "binned"),
    )
    fig.update_annotations(font_size=10)  # set font size of subplot titles
    # plot content
    is_numeric = trn_num_kde is not None
    if is_numeric:
        trn_line1, syn_line1 = plot_univariate_distribution_numeric(trn_num_kde, syn_num_kde)
        trn_line2, syn_line2 = plot_univariate_binned(trn_bin_col_cnts, syn_bin_col_cnts, sort_by_frequency=False)
        # prevent Plotly from trying to convert strings to dates
        fig.layout.xaxis2.update(type="category")
    else:
        fig.layout.yaxis.update(tickformat=".0%")
        trn_line1, syn_line1 = plot_univariate_distribution_categorical(trn_cat_col_cnts, syn_cat_col_cnts)
        trn_line2, syn_line2 = plot_univariate_binned(trn_bin_col_cnts, syn_bin_col_cnts, sort_by_frequency=True)
        # prevent Plotly from trying to convert strings to dates
        fig.layout.xaxis.update(type="category")
        fig.layout.xaxis2.update(type="category")

    # rescale y2 axis dependent on max peak
    y_max = min(0.999, 2.0 * max(trn_line2["y"]))
    fig.layout.yaxis2.update(range=[0, y_max], tickformat=".0%")

    fig.add_trace(trn_line1, row=1, col=1)
    fig.add_trace(syn_line1, row=1, col=1)
    fig.add_trace(trn_line2, row=1, col=2)
    fig.add_trace(syn_line2, row=1, col=2)
    return fig


def prepare_categorical_plot_data_distribution(
    trn_col_cnts: pd.Series,
    syn_col_cnts: pd.Series,
) -> pd.DataFrame:
    trn_col_cnts_idx = trn_col_cnts.index.to_series().astype("string").fillna(NA_BIN).replace("", EMPTY_BIN)
    syn_col_cnts_idx = syn_col_cnts.index.to_series().astype("string").fillna(NA_BIN).replace("", EMPTY_BIN)
    trn_col_cnts = trn_col_cnts.set_axis(trn_col_cnts_idx)
    syn_col_cnts = syn_col_cnts.set_axis(syn_col_cnts_idx)
    t = trn_col_cnts.to_frame("target_cnt").reset_index()
    s = syn_col_cnts.to_frame("synthetic_cnt").reset_index()
    df = pd.merge(t, s, on="index", how="outer")
    df["target_cnt"] = df["target_cnt"].fillna(0.0)
    df["synthetic_cnt"] = df["synthetic_cnt"].fillna(0.0)
    df["avg_cnt"] = (df["target_cnt"] + df["synthetic_cnt"]) / 2
    df = df[df["avg_cnt"] > 0]
    df["target_pct"] = df["target_cnt"] / df["target_cnt"].sum()
    df["synthetic_pct"] = df["synthetic_cnt"] / df["synthetic_cnt"].sum()
    df = df.rename(columns={"index": "category"})
    if df["category"].dtype.name == "category":
        df["category_code"] = df["category"].cat.codes
    else:
        df["category_code"] = df["category"]
    df = df.sort_values("category_code", ascending=True).reset_index(drop=True)
    return df


def prepare_categorical_plot_data_binned(
    trn_bin_col_cnts: pd.Series,
    syn_bin_col_cnts: pd.Series,
    sort_by_frequency: bool,
) -> pd.DataFrame:
    t = trn_bin_col_cnts.to_frame("target_cnt").reset_index(names="category")
    s = syn_bin_col_cnts.to_frame("synthetic_cnt").reset_index(names="category")
    df = pd.merge(t, s, on="category", how="outer")
    df["target_cnt"] = df["target_cnt"].fillna(0.0)
    df["synthetic_cnt"] = df["synthetic_cnt"].fillna(0.0)
    df["avg_cnt"] = (df["target_cnt"] + df["synthetic_cnt"]) / 2
    df = df[df["avg_cnt"] > 0]
    df["target_pct"] = df["target_cnt"] / df["target_cnt"].sum()
    df["synthetic_pct"] = df["synthetic_cnt"] / df["synthetic_cnt"].sum()
    if df["category"].dtype.name == "category":
        df["category_code"] = df["category"].cat.codes
    else:
        df["category_code"] = df["category"]
    if sort_by_frequency:
        df = df.sort_values("target_pct", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("category_code", ascending=True).reset_index(drop=True)
    return df


def plot_univariate_distribution_categorical(
    trn_cat_col_cnts: pd.Series, syn_cat_col_cnts: pd.Series
) -> tuple[go.Scatter, go.Scatter]:
    # prepare data
    df = prepare_categorical_plot_data_distribution(trn_cat_col_cnts, syn_cat_col_cnts)
    df = df.sort_values("avg_cnt", ascending=False)
    # trim labels
    df["category"] = trim_labels(df["category"], max_length=10)
    # prepare plots
    trn_line = go.Scatter(
        mode="lines",
        x=df["category"],
        y=df["target_pct"],
        name="original",
        line_color=CHARTS_COLORS["original"],
        yhoverformat=".2%",
    )
    syn_line = go.Scatter(
        mode="lines",
        x=df["category"],
        y=df["synthetic_pct"],
        name="synthetic",
        line_color=CHARTS_COLORS["synthetic"],
        yhoverformat=".2%",
        fill="tonexty",
        fillcolor=CHARTS_COLORS["gap"],
    )
    return trn_line, syn_line


def plot_univariate_binned(
    trn_bin_col_cnts: pd.Series,
    syn_bin_col_cnts: pd.Series,
    sort_by_frequency: bool = False,
) -> tuple[go.Scatter, go.Scatter]:
    # prepare data
    df = prepare_categorical_plot_data_binned(trn_bin_col_cnts, syn_bin_col_cnts, sort_by_frequency)
    # prepare plots
    trn_line = go.Scatter(
        mode="lines+markers",
        x=df["category"],
        y=df["target_pct"],
        name="original",
        line_color=CHARTS_COLORS["original"],
        yhoverformat=".2%",
        marker_symbol="diamond",
        marker_size=6,
    )
    syn_line = go.Scatter(
        mode="lines+markers",
        x=df["category"],
        y=df["synthetic_pct"],
        name="synthetic",
        line_color=CHARTS_COLORS["synthetic"],
        yhoverformat=".2%",
        fill="tonexty",
        fillcolor=CHARTS_COLORS["gap"],
        marker_symbol="diamond",
        marker_size=6,
    )
    return trn_line, syn_line


def plot_univariate_distribution_numeric(
    trn_num_kde: pd.Series, syn_num_kde: pd.Series
) -> tuple[go.Scatter, go.Scatter]:
    trn_line = go.Scatter(
        x=trn_num_kde.index,
        y=trn_num_kde.values,
        name="original",
        line_color=CHARTS_COLORS["original"],
        yhoverformat=".5f",
    )
    syn_line = go.Scatter(
        x=syn_num_kde.index,
        y=syn_num_kde.values,
        name="synthetic",
        line_color=CHARTS_COLORS["synthetic"],
        yhoverformat=".5f",
        fill="tonexty",
        fillcolor=CHARTS_COLORS["gap"],
    )
    return trn_line, syn_line


def plot_store_bivariates(
    trn_cnts_uni: dict[str, pd.Series],
    syn_cnts_uni: dict[str, pd.Series],
    trn_cnts_biv: dict[tuple[str, str], pd.Series],
    syn_cnts_biv: dict[tuple[str, str], pd.Series],
    acc_biv: pd.DataFrame,
    workspace: TemporaryWorkspace,
    show_accuracy: bool,
) -> None:
    """
    Plots all bivariate accuracy figures and stores them under workspace dir.
    """

    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        Parallel()(
            delayed(plot_store_bivariate)(
                row["col1"],
                row["col2"],
                trn_cnts_uni[row["col1"]],
                trn_cnts_uni[row["col2"]],
                syn_cnts_uni[row["col1"]],
                syn_cnts_uni[row["col2"]],
                trn_cnts_biv[(row["col1"], row["col2"])],
                syn_cnts_biv[(row["col1"], row["col2"])],
                row["accuracy"] if show_accuracy else None,
                workspace,
            )
            for _, row in acc_biv.iterrows()
        )


def plot_store_bivariate(
    col1: str,
    col2: str,
    trn_cnts_col1: pd.Series,
    trn_cnts_col2: pd.Series,
    syn_cnts_col1: pd.Series,
    syn_cnts_col2: pd.Series,
    trn_cnts_col12: pd.Series,
    syn_cnts_col12: pd.Series,
    accuracy: float | None,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_bivariate(
        col1,
        col2,
        trn_cnts_col1,
        trn_cnts_col2,
        syn_cnts_col1,
        syn_cnts_col2,
        trn_cnts_col12,
        syn_cnts_col12,
        accuracy,
    )
    workspace.store_figure_html(fig, "bivariate", col1, col2)


def plot_bivariate(
    col1: str,
    col2: str,
    trn_cnts_col1: pd.Series,
    trn_cnts_col2: pd.Series,
    syn_cnts_col1: pd.Series,
    syn_cnts_col2: pd.Series,
    trn_cnts_col12: pd.Series,
    syn_cnts_col12: pd.Series,
    accuracy: float | None,
) -> go.Figure:
    # prepare data
    # establish grid of cross-combinations
    x = pd.concat([trn_cnts_col1.index.to_series(), syn_cnts_col1.index.to_series()]).drop_duplicates().to_frame(col1)
    y = pd.concat([trn_cnts_col2.index.to_series(), syn_cnts_col2.index.to_series()]).drop_duplicates().to_frame(col2)
    df = pd.merge(x, y, how="cross")
    df = pd.merge(
        df,
        trn_cnts_col12.to_frame("target").reset_index(),
        how="left",
    )
    df = pd.merge(
        df,
        syn_cnts_col12.to_frame("synthetic").reset_index(),
        how="left",
    )
    df = df.reset_index(drop=True)
    df["target"] = df["target"].fillna(0.0)
    df["synthetic"] = df["synthetic"].fillna(0.0)
    # normalize values row-wise (used for hover)
    df["target_by_row"] = df["target"] / df.groupby(col1, observed=False)["target"].transform("sum")
    df["synthetic_by_row"] = df["synthetic"] / df.groupby(col1, observed=False)["synthetic"].transform("sum")
    # normalize values across table (used for visualization + accuracy)
    df["target_by_all"] = df["target"] / df["target"].sum()
    df["synthetic_by_all"] = df["synthetic"] / df["synthetic"].sum()
    # round displayed numerics to reduce HTML size
    df["target_by_row"] = df["target_by_row"].round(5)
    df["synthetic_by_row"] = df["synthetic_by_row"].round(5)
    df["target_by_all"] = df["target_by_all"].round(5)
    df["synthetic_by_all"] = df["synthetic_by_all"].round(5)
    df["y"] = df[col1].astype("str")
    df["x"] = df[col2].astype("str")
    # plot title
    col1_name = trim_label(col1, max_length=30)
    col2_name = trim_label(col2, max_length=30)
    plot_title = f"<b>{col1_name} ~ {col2_name}</b>" + (f" <sup>{accuracy:.1%}</sup>" if accuracy is not None else "")
    # plot layout
    layout = go.Layout(
        title=dict(text=plot_title, x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=220,
        margin=dict(l=10, r=10, b=10, t=40, pad=5),
        showlegend=False,
        # prevent Plotly from trying to convert strings to dates
        xaxis=dict(type="category"),
        xaxis2=dict(type="category"),
        yaxis=dict(type="category"),
        yaxis2=dict(type="category"),
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.05,
        shared_yaxes=True,
        subplot_titles=("original", "synthetic"),
    )
    fig.update_annotations(font_size=10)  # set font size of subplot titles
    # plot content
    hovertemplate = col1_name[:10] + ": `%{y}`<br />" + col2_name[:10] + ": `%{x}`<br /><br />"
    hovertemplate += "share original vs. synthetic<br />"
    hovertemplate += "row-wise: %{customdata[0]} vs. %{customdata[1]}<br />"
    hovertemplate += "absolute: %{customdata[2]} vs. %{customdata[3]}<br />"
    customdata = df[["target_by_row", "synthetic_by_row", "target_by_all", "synthetic_by_all"]].apply(
        lambda x: x.map("{:.2%}".format)
    )
    heat1 = go.Heatmap(
        x=df["x"],
        y=df["y"],
        z=df["target_by_row"],
        name="target",
        zmin=0,
        zmax=1.0,
        customdata=customdata,
        autocolorscale=False,
        colorscale=["white", "#A7A7A7", "#7B7B7B", "#666666"],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    heat2 = go.Heatmap(
        x=df["x"],
        y=df["y"],
        z=df["synthetic_by_row"],
        name="synthetic",
        zmin=0,
        zmax=1.0,
        customdata=customdata,
        autocolorscale=False,
        colorscale=["white", "#81EAC3", "#43E0A5", "#24DB96"],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    fig.add_trace(heat1, row=1, col=1)
    fig.add_trace(heat2, row=1, col=2)
    return fig


def plot_store_accuracy_matrix(
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    """
    Plots accuracy matrix and stores it under workspace dir.
    """

    fig = plot_accuracy_matrix(acc_uni, acc_biv)
    workspace.store_figure_html(fig, "accuracy_matrix")


def plot_accuracy_matrix(acc_uni: pd.DataFrame, acc_biv: pd.DataFrame) -> go.Figure:
    # prepare data
    acc_df = pd.concat(
        ([acc_biv[acc_biv.type == "tgt"]] if not acc_biv.empty else [])
        + [acc_uni.assign(col1=acc_uni.column).assign(col2=acc_uni.column)],
        axis=0,
    ).reset_index(drop=True)
    acc_mat = acc_df.pivot(index="col1", columns="col2", values="accuracy")
    # plot layout
    layout = go.Layout(
        title=dict(text="<b>Accuracy Matrix</b>", x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=500,
        margin=dict(l=10, r=10, b=10, t=30, pad=5),
        showlegend=False,
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=1,
    )
    # plot content
    col_names = trim_labels(acc_mat.columns, max_length=30)
    hovertemplate = "`%{x}` vs. `%{y}`: %{z:.2%}"
    heat = go.Heatmap(
        x=col_names,
        y=col_names,
        z=acc_mat,
        name="accuracy",
        autocolorscale=False,
        zmin=0.8,
        zmax=1,
        colorscale=["#F5F6FF", "#3D4FFF"],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    fig.add_trace(heat, row=1, col=1)
    return fig


def format_display_prefixes(*labels: str) -> list[str]:
    """
    Reformats labels with "ctx:", "tgt:" and "nxt:" prefixes
    to their displayable form.
    """

    prefix_display_replacements = {
        CTX_COLUMN_PREFIX: "context:",
        TGT_COLUMN_PREFIX: "",
        NXT_COLUMN_PREFIX: "",
    }

    def _format_for_display(label: str) -> str:
        label = str(label)
        prefix = next(
            (prefix for prefix in prefix_display_replacements if label.startswith(prefix)),
            None,
        )
        replacement = prefix_display_replacements.get(prefix)
        return label.replace(prefix, replacement, 1) if replacement is not None else label

    return [_format_for_display(n) for n in labels]


def trim_label(label: str, max_length: int = None, reserved_labels: set[str] = None) -> str:
    if reserved_labels is None:
        reserved_labels = set()

    [label] = format_display_prefixes(label)

    def truncate_middle(s, n):
        if len(s) <= n:
            return s  # string is already short-enough
        n -= 3  # three dots replace text
        n_1 = math.ceil(n / 2)
        n_2 = math.floor(n / 2)
        return "{0}...{1}".format(s[:n_1], s[-n_2:])

    if max_length is not None:
        label = truncate_middle(label, max_length)
        if label in reserved_labels:
            idx = 0
            while True:
                unique_label = label + str(idx)
                if unique_label not in reserved_labels:
                    break
                idx += 1
            label = unique_label

    return label


def trim_labels(labels: list[str], max_length: int = None, ensure_unique=True) -> list[str]:
    out = []
    trimmed_labels = set()
    for i, label in enumerate(labels):
        trimmed_label = trim_label(
            label,
            max_length=max_length,
            reserved_labels=trimmed_labels if ensure_unique else None,
        )
        out.append(trimmed_label)
        trimmed_labels.add(trimmed_label)
    return out


def binning_data(
    trn: pd.DataFrame,
    syn: pd.DataFrame,
    statistics: Statistics,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _LOG.info("calculate original data bins")
    trn_bin, bins = bin_data(df=trn, bins=10)
    _LOG.info("store original data bins")
    statistics.store_bins(bins=bins)
    _LOG.info("calculate synthetic data bins")
    syn_bin, _ = bin_data(df=syn, bins=bins)
    return trn_bin, syn_bin


def bin_data(df: pd.DataFrame, bins: int | dict[str, list]) -> tuple[pd.DataFrame, dict[str, list]]:
    """
    Splits data into bins.
    Binning algorithm depends on column type. Categorical binning creates 'n' bins corresponding to the highest
    cardinality categories and so-called '(other)' bin for all remaining categories. Numerical binning attempts to
    create 'n' equally-sized bins and so-called '(n/a)' bin for missing values. Bins can also be provided as a
    dictionary of column names and lists of bin boundaries. In this case, binning boundaries search is skipped and
    bin boundaries are used as is. Regardless of binning strategy, bin boundaries are calculated with respect to
    training data and synthetic data is mapped accordingly.
    """

    # Note, that we create a new pd.DataFrame to avoid fragmentation warning messages that can occur if we try to
    # replace hundreds of columns of a large dataset
    cols = {}

    bins_dct = {}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dat_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols + dat_cols]
    if isinstance(bins, int):
        for col in num_cols:
            cols[col], bins_dct[col] = bin_numeric(df[col], bins)
        for col in dat_cols:
            cols[col], bins_dct[col] = bin_datetime(df[col], bins)
        for col in cat_cols:
            cols[col], bins_dct[col] = bin_categorical(df[col], bins)
    else:  # bins is a dict
        for col in num_cols:
            if col in bins:
                cols[col], _ = bin_numeric(df[col], bins[col])
            else:
                _LOG.warning(f"'{col}' is missing in bins")
        for col in dat_cols:
            if col in bins:
                cols[col], _ = bin_datetime(df[col], bins[col])
            else:
                _LOG.warning(f"'{col}' is missing in bins")
        for col in cat_cols:
            if col in bins:
                cols[col], _ = bin_categorical(df[col], bins[col])
            else:
                _LOG.warning(f"'{col}' is missing in bins")
        bins_dct = bins
    return pd.DataFrame(cols), bins_dct


def bin_numeric(col: pd.Series, bins: int | list[str]) -> tuple[pd.Categorical, list]:
    def _clip(col, bins):
        if isinstance(bins, list):
            # use precomputed bin boundaries
            if len(bins) > 0:
                return col.clip(min(bins), max(bins)), bins
            else:
                return col, bins

        if col.nunique() <= bins:
            max_val = pd.concat([col]).max()
            upper_limit = [max_val + 1] if not pd.isna(max_val) else []
            breaks = sorted(col.dropna().unique()) + upper_limit
        else:
            breaks = search_bin_boundaries(col, bins)
        # ensure that we have unique breaks in increasing order
        breaks = list(sorted(set(breaks)))
        if len(breaks) == 0:
            breaks = [-1e10, 1e10]

        col = col.clip(min(breaks), max(breaks))

        def _floor(number, precision):
            return np.true_divide(np.floor(number * 10**precision), 10**precision)

        precisions = list(range(20))
        for precision in precisions:
            if len(set([_floor(b, precision) for b in breaks])) == len(breaks):
                breaks = [_floor(b, precision) for b in breaks[:-1]] + [breaks[-1]]
                break

        return col, breaks

    def _define_labels(breaks):
        labels = breaks[:-1]
        if all(x.is_integer() for x in labels):
            labels = [int(label) for label in labels]
        labels = [str(label) for label in labels]

        return labels

    def _adjust_breaks(breaks):
        return breaks[:-1] + [breaks[-1] + 1]

    return bin_non_categorical(col, bins, _clip, _define_labels, _adjust_breaks)


def bin_datetime(col: pd.Series, bins: int | list[str]) -> tuple[pd.Categorical, list]:
    def _clip(col, bins):
        if isinstance(bins, list):
            # use precomputed bin boundaries
            if len(bins) > 0:
                return col.clip(min(bins), max(bins)), bins
            else:
                return col, bins

        if col.nunique() == 1:
            # ensure 2 breaks for single-valued columns
            val = col.iloc[0]
            upper_limit = [val + np.timedelta64(1, "D")] if not pd.isna(val) else []
            breaks = [val] + upper_limit
        else:
            breaks = search_bin_boundaries(col, bins)
        # ensure that we have unique breaks in increasing order
        breaks = list(sorted(set(breaks)))
        if len(breaks) == 0:
            breaks = pd.to_datetime(["1700", "2200"]).to_list()

        col = col.clip(min(breaks), max(breaks))
        return col, breaks

    def _define_labels(breaks):
        labels = breaks[:-1]
        formats = [
            "%Y",
            "%Y-%b",
            "%Y-%b-%d",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]
        for label_format in formats:
            if len(set(pd.to_datetime(breaks).strftime(label_format))) == len(breaks):
                labels = [pd.to_datetime(b).strftime(label_format) for b in labels]
                break
        return labels

    def _adjust_breaks(breaks):
        return breaks[:-1] + [max(breaks[-1] + np.timedelta64(1, "D"), breaks[-1])]

    return bin_non_categorical(col, bins, _clip, _define_labels, _adjust_breaks)


def bin_non_categorical(
    col: pd.Series,
    bins: int | list,
    clip_and_breaks: Callable,
    create_labels: Callable,
    adjust_breaks: Callable,
) -> tuple[pd.Categorical, list]:
    col = col.fillna(np.nan).infer_objects(copy=False)

    col, breaks = clip_and_breaks(col, bins)
    labels = create_labels(breaks)

    if not len(labels) == len(set(labels)) or len(labels) + 1 != len(breaks):
        # keep the invariants that labels need to be unique and there always
        # must be strictly one less labels than breaks before using pd.cut
        # breaks is assumed to be unique by this point
        _LOG.warning(
            "one of labels invariants is broken in binning; "
            f"not len(labels) == len(set(labels)): {not len(labels) == len(set(labels))}; "
            f"len(labels) + 1 != len(breaks): {len(labels) + 1 != len(breaks)}; "
            "falling back to sensible default for labels"
        )
        labels = [str(b) for b in breaks[:-1]]

    new_labels_map = {label: f"⪰ {label}" for label in labels}

    bin_col = pd.cut(col, bins=adjust_breaks(breaks), labels=labels, right=False)
    bin_col = bin_col.cat.rename_categories(new_labels_map)
    bin_col = bin_col.values.add_categories(NA_BIN).fillna(NA_BIN)

    return bin_col, breaks


def bin_categorical(col: pd.Series, bins: int | list[str]) -> tuple[pd.Categorical, list[str]]:
    col = col.fillna(NA_BIN)
    col = col.replace("", EMPTY_BIN)
    # determine top values, if not provided
    # and ensure that privacy is protected
    if isinstance(bins, int):
        cnts = col.value_counts().head(bins)
        cnts = cnts[cnts >= MIN_RARE_CAT_PROTECTION]
        bins = sorted(list(cnts.index))
    # shift special categories to last position
    bins = [c for c in bins if c != EMPTY_BIN] + [EMPTY_BIN]
    bins = [c for c in bins if c != OTHER_BIN] + [OTHER_BIN]
    bins = [c for c in bins if c != RARE_BIN] + [RARE_BIN]
    bins = [c for c in bins if c != NA_BIN] + [NA_BIN]
    col = pd.Categorical(col, categories=bins, ordered=True)
    col = col.fillna(OTHER_BIN)
    new_cats = dict(zip(bins, trim_labels(bins, max_length=20)))
    col = col.rename_categories(new_cats)
    return col, bins


def search_bin_boundaries(num_col: pd.Series, n: int) -> list:
    # greedily search for `n` distinct bucket boundaries; includes min and max values
    values = np.sort(num_col.dropna())
    if len(values) == 0:
        return []
    breaks = []
    search_start = n
    search_end = max(n + 1, min(1000, len(values) + 1))
    for i in range(search_start, search_end):
        indices = np.linspace(0, len(values) - 1, i).astype(int)
        breaks = sorted(list(set(values[indices])))
        if len(breaks) >= n + 1:
            break
    return breaks


def plot_store_correlation_matrices(
    corr_trn: pd.DataFrame,
    corr_syn: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    """
    Plots correlation matrices for target and synthetic data, plus
    matrix representing difference between the two and stores them
    under workspace dir.
    """

    fig = plot_correlation_matrices(corr_trn, corr_syn)
    workspace.store_figure_html(fig, "correlation_matrices")


def calculate_correlations(binned: pd.DataFrame, corr_cols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Calculates correlations between target columns.
    'phik' library is used to calculate correlation matrices,
    which are then ordered with respect to
    hierarchical linkage based on training data.
    """

    tgt_cols = [c for c in binned.columns if c.startswith(TGT_COLUMN_PREFIX)]
    # calculate correlation matrices
    corr = phik.phik.phik_from_rebinned_df(binned[tgt_cols].copy())
    # constant columns result in NAs, that we replace with zeros
    corr = corr.fillna(0)
    # trim superfluous precision
    corr = corr.round(5)

    # re-order correlation matrix
    if corr_cols is None:
        try:
            # determine column order via hierarchical linkage based on data; this mimics the logic used within
            # `seaborn.clustermap`. We wrap this in try/except to be on the safe side, as `dendogram` and `linkage` can
            # raise errors for edge cases. And because the improved sort order is not critical to have in place.
            corr_link = fastcluster.linkage(corr, method="complete", metric="euclidean")
            corr_idx = scipy.cluster.hierarchy.dendrogram(corr_link, no_plot=True)["leaves"]
            corr_cols = corr.columns[corr_idx]
        except Exception:
            corr_cols = corr.columns
    corr = corr.reindex(corr_cols, axis=0).reindex(corr_cols, axis=1)
    return corr


def plot_correlation_matrices(corr_trn: pd.DataFrame, corr_syn: pd.DataFrame) -> go.Figure:
    # plot layout
    layout = go.Layout(
        title=dict(text="<b>Correlation Matrices</b>", x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=400,
        margin=dict(l=10, r=10, b=10, t=50, pad=5),
        showlegend=False,
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.05,
        shared_yaxes=True,
        subplot_titles=("original", "synthetic", "difference"),
    )
    fig.update_annotations(font_size=12)  # set font size of subplot titles
    # plot content
    col_names = trim_labels(corr_trn.columns, max_length=30)
    hovertemplate = "`%{x}` vs. `%{y}`: %{z:.2f}"
    heat1 = go.Heatmap(
        x=col_names,
        y=col_names,
        z=corr_trn,
        name="original",
        zmin=0,
        zmax=1.0,
        autocolorscale=False,
        colorscale=["white", CHARTS_COLORS["original"]],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    heat2 = go.Heatmap(
        x=col_names,
        y=col_names,
        z=corr_syn,
        name="synthetic",
        zmin=0,
        zmax=1.0,
        autocolorscale=False,
        colorscale=["white", CHARTS_COLORS["synthetic"]],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    heat3 = go.Heatmap(
        x=col_names,
        y=col_names,
        z=(corr_trn - corr_syn).abs(),
        name="difference",
        zmin=0,
        zmax=1.0,
        autocolorscale=False,
        colorscale=["white", CHARTS_COLORS["difference"]],
        showscale=False,
        hovertemplate=hovertemplate,
    )
    fig.add_trace(heat1, row=1, col=1)
    fig.add_trace(heat2, row=1, col=2)
    fig.add_trace(heat3, row=1, col=3)
    return fig


def filter_head_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.shape[0] > n:
        return pd.concat([df.head(n // 2), df.tail(n // 2)], axis=0)
    else:
        return df


def filter_uni_acc_for_plotting(acc_uni: pd.DataFrame) -> pd.DataFrame:
    # limit displayed univariate charts; with half being most accurate, and half being least accurate
    acc_uni = acc_uni.sort_values("accuracy", ascending=False)
    acc_uni = filter_head_tail(acc_uni, MAX_UNIVARIATE_PLOTS)
    return acc_uni.reset_index(drop=True)


def filter_biv_acc_for_plotting(acc_biv: pd.DataFrame, corr_trn: pd.DataFrame | None) -> pd.DataFrame:
    if corr_trn is not None:
        # enrich acc_biv with calculated correlations from original data
        acc_biv = (
            corr_trn.reset_index()
            .rename(columns={"index": "col1"})
            .melt(id_vars="col1", var_name="col2", value_name="correlation")
            .merge(acc_biv, on=["col1", "col2"], how="right")
        )
        # sort bivariates by strength of correlation, as these are the most interesting patterns
        acc_biv = acc_biv.sort_values(["correlation", "accuracy"], ascending=False)
    # take only tgt:col1 ~ tgt:col2 (and not tgt:col2 ~ tgt:col1)
    acc_biv_tgt = acc_biv.loc[acc_biv.type == "tgt"]
    acc_biv_tgt = acc_biv_tgt.loc[acc_biv_tgt.col1 < acc_biv_tgt.col2]
    acc_biv_tgt = filter_head_tail(acc_biv_tgt, MAX_BIVARIATE_TGT_PLOTS)
    # take only ctx:col ~ tgt:col (and not tgt:col ~ ctx:col)
    acc_biv_ctx = acc_biv.loc[acc_biv.type == "ctx"]
    acc_biv_ctx = acc_biv_ctx.loc[acc_biv_ctx.col1.str.startswith("ctx")]
    acc_biv_ctx = filter_head_tail(acc_biv_ctx, MAX_BIVARIATE_CTX_PLOTS)
    # take only tgt:col ~ nxt:col (and not nxt:col ~ tgt:col)
    acc_biv_nxt = acc_biv.loc[acc_biv.type == "nxt"]
    acc_biv_nxt = acc_biv_nxt.loc[acc_biv_nxt.col2.str.startswith("nxt")]
    acc_biv_nxt = filter_head_tail(acc_biv_nxt, MAX_BIVARIATE_NXT_PLOTS)
    # concatenate all together
    acc = pd.concat([acc_biv_tgt, acc_biv_ctx, acc_biv_nxt]).reset_index(drop=True)
    return acc.reset_index(drop=True)
