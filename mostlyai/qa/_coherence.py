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

from joblib import Parallel, cpu_count, delayed, parallel_config
import pandas as pd
from plotly import graph_objs as go

from mostlyai.qa._accuracy import (
    bin_data,
    calculate_accuracy,
    plot_univariate,
    prepare_categorical_plot_data_distribution,
    trim_label,
    trim_labels,
)
from mostlyai.qa._common import CHARTS_COLORS, CHARTS_FONTS
from mostlyai.qa._filesystem import TemporaryWorkspace
from mostlyai.qa._sampling import harmonize_dtype


def plot_store_coherences(
    trn_num_kdes: dict[str, pd.Series],
    syn_num_kdes: dict[str, pd.Series],
    trn_cat_cnts: dict[str, pd.Series],
    syn_cat_cnts: dict[str, pd.Series],
    acc_coh: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        Parallel()(
            delayed(plot_store_coherence)(
                row["column"],
                trn_num_kdes.get(row["column"]),
                syn_num_kdes.get(row["column"]),
                trn_cat_cnts.get(row["column"]),
                syn_cat_cnts.get(row["column"]),
                row["accuracy"],
                workspace,
            )
            for _, row in acc_coh.iterrows()
        )


def plot_store_coherence(
    col: str,
    trn_num_kde: pd.Series | None,
    syn_num_kde: pd.Series | None,
    trn_cat_cnts: pd.Series | None,
    syn_cat_cnts: pd.Series | None,
    accuracy: float | None,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_coherence(
        col,
        trn_num_kde,
        syn_num_kde,
        trn_cat_cnts,
        syn_cat_cnts,
        accuracy,
    )
    workspace.store_figure_html(fig, "coherence", col)


def plot_coherence(
    col_name: str,
    trn_num_kde: pd.Series | None,
    syn_num_kde: pd.Series | None,
    trn_cat_cnts: pd.Series | None,
    syn_cat_cnts: pd.Series | None,
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
        # prevent Plotly from trying to convert strings to dates
        fig.layout.xaxis2.update(type="category")
    else:
        fig.layout.yaxis.update(tickformat=".0%")
        trn_line1, syn_line1 = plot_univariate_distribution_categorical(trn_cat_cnts, syn_cat_cnts)
        # prevent Plotly from trying to convert strings to dates
        fig.layout.xaxis.update(type="category")
        fig.layout.xaxis2.update(type="category")

    # rescale y2 axis dependent on max peak
    # y_max = min(0.999, 2.0 * max(trn_line2["y"]))
    # fig.layout.yaxis2.update(range=[0, y_max], tickformat=".0%")

    fig.add_trace(trn_line1, row=1, col=1)
    fig.add_trace(syn_line1, row=1, col=1)
    # fig.add_trace(trn_line2, row=1, col=2)
    # fig.add_trace(syn_line2, row=1, col=2)
    return fig


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


def pull_data_for_coherence(
    *,
    df_tgt: pd.DataFrame,
    tgt_context_key: str,
    max_sequence_length: int = 100,
) -> pd.DataFrame:
    """
    Prepare sequential dataset for coherence metrics.
    """
    df_tgt = df_tgt.copy()

    # randomly sample at most max_sequence_length rows per sequence
    df_tgt = df_tgt.sample(frac=1).reset_index(drop=True)
    df_tgt = df_tgt[df_tgt.groupby(tgt_context_key).cumcount() < max_sequence_length].reset_index(drop=True)

    # harmonize dtypes
    # apply harmonize_dtype to all columns except tgt_context_key
    df_tgt = df_tgt.apply(lambda col: harmonize_dtype(col) if col.name != tgt_context_key else col)

    # harmonize categorical columns
    num_cols = [c for c in df_tgt.columns if c != tgt_context_key and pd.api.types.is_numeric_dtype(df_tgt[c])]
    dat_cols = [c for c in df_tgt.columns if c != tgt_context_key and pd.api.types.is_datetime64_any_dtype(df_tgt[c])]
    for col in df_tgt.columns:
        if col == tgt_context_key or col in num_cols + dat_cols:
            continue
        df_tgt[col] = pd.Categorical(df_tgt[col], ordered=True)

    # discretize datetime and numeric columns
    if num_cols or dat_cols:
        binned_df, _ = bin_data(df_tgt[num_cols + dat_cols], bins=30)
        for col in num_cols + dat_cols:
            df_tgt[col] = binned_df[col]

    # replace all null values with '(n/a)'
    df_tgt = df_tgt.copy()
    for col in df_tgt.columns:
        if col == tgt_context_key:
            continue
        # Add '(n/a)' category if needed and replace nulls
        if df_tgt[col].isna().any():
            df_tgt[col] = df_tgt[col].cat.add_categories("(n/a)")
            df_tgt.loc[df_tgt[col].isna(), col] = "(n/a)"

    # TODO: update example output
    # Example output (pd.DataFrame):
    # | players_id | year   | team | league | G    | AB    | R    | H    | HR   | RBI  | SB   | CS   | BB   | SO   |
    # |------------|--------|------|--------|------|-------|------|------|------|------|------|------|------|------|
    # | borowha01  | 1943.0 | NYA  | AL     | 29.0 | 74.0  | 2.0  | 15.0 | 0.0  | 7.0  | 0.0  | 0.0  | 5.0  | 17.0 |
    # | wallaja02  | 1946.0 | PHA  | AL     | 63.0 | 194.0 | 16.0 | 38.0 | 5.0  | 11.0 | 1.0  | 0.0  | 14.0 | 47.0 |
    # players_id dtype: original, other columns dtype: Categorical
    return df_tgt


def calculate_categories_per_sequence(df: pd.DataFrame, context_key: str) -> pd.DataFrame:
    """
    Calculate the number of categories per sequence for all columns except the context key.
    """
    # Example output (pd.DataFrame):
    # | year | team | league | G  | AB | R  | H  | HR | RBI | SB | CS | BB | SO |
    # |------|------|--------|----|----|----|----|----|-----|----|----|----|----|
    # | 9    | 8    | 2      | 9  | 3  | 1  | 1  | 1  | 1   | 1  | 1  | 1  | 2  |
    # | 23   | 3    | 2      | 18 | 21 | 20 | 23 | 17 | 20  | 15 | 10 | 22 | 19 |
    # columns dtype: int64
    return df.groupby(context_key).nunique().reset_index(drop=True)


def calculate_categories_per_sequence_accuracy(
    cats_per_seq_trn_binned: pd.DataFrame, cats_per_seq_syn_binned: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate accuracy of categories per sequence.
    """
    acc_cats_per_seq = pd.DataFrame({"column": cats_per_seq_trn_binned.columns})
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        results = Parallel()(
            delayed(calculate_accuracy)(
                trn_bin_cols=cats_per_seq_trn_binned[[row["column"]]],
                syn_bin_cols=cats_per_seq_syn_binned[[row["column"]]],
            )
            for _, row in acc_cats_per_seq.iterrows()
        )
        acc_cats_per_seq["accuracy"], acc_cats_per_seq["accuracy_max"] = zip(*results)
    return acc_cats_per_seq


def plot_store_categories_per_sequence(
    cats_per_seq_trn_kdes: dict[str, pd.Series],
    cats_per_seq_syn_kdes: dict[str, pd.Series],
    cats_per_seq_trn_binned_cnts: dict[str, pd.Series],
    cats_per_seq_syn_binned_cnts: dict[str, pd.Series],
    acc_cats_per_seq: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        Parallel()(
            delayed(plot_store_single_categories_per_sequence)(
                row["column"],
                cats_per_seq_trn_kdes.get(row["column"]),
                cats_per_seq_syn_kdes.get(row["column"]),
                cats_per_seq_trn_binned_cnts.get(row["column"]),
                cats_per_seq_syn_binned_cnts.get(row["column"]),
                row["accuracy"],
                workspace,
            )
            for _, row in acc_cats_per_seq.iterrows()
        )


def plot_store_single_categories_per_sequence(
    col: str,
    cats_per_seq_trn_kde: pd.Series,
    cats_per_seq_syn_kde: pd.Series,
    cats_per_seq_trn_binned_cnt: pd.Series,
    cats_per_seq_syn_binned_cnt: pd.Series,
    accuracy: float,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_categories_per_sequence(
        col=col,
        cats_per_seq_trn_kde=cats_per_seq_trn_kde,
        cats_per_seq_syn_kde=cats_per_seq_syn_kde,
        cats_per_seq_trn_binned_cnt=cats_per_seq_trn_binned_cnt,
        cats_per_seq_syn_binned_cnt=cats_per_seq_syn_binned_cnt,
        accuracy=accuracy,
    )
    workspace.store_figure_html(fig, "categories_per_sequence", col)


def plot_categories_per_sequence(
    col: str,
    cats_per_seq_trn_kde: pd.Series,
    cats_per_seq_syn_kde: pd.Series,
    cats_per_seq_trn_binned_cnt: pd.Series,
    cats_per_seq_syn_binned_cnt: pd.Series,
    accuracy: float,
) -> go.Figure:
    return plot_univariate(
        col_name=col,
        trn_num_kde=cats_per_seq_trn_kde,
        syn_num_kde=cats_per_seq_syn_kde,
        trn_cat_col_cnts=None,
        syn_cat_col_cnts=None,
        trn_bin_col_cnts=cats_per_seq_trn_binned_cnt,
        syn_bin_col_cnts=cats_per_seq_syn_binned_cnt,
        accuracy=accuracy,
    )


def calculate_sequences_per_category(
    df: pd.DataFrame, context_key: str
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], int]:
    """
    Calculate the number of sequences per category for all columns except the context key.
    """
    sequences_per_category_dict = {
        col: df.groupby(col)[context_key].nunique().rename_axis(None) for col in df.columns if col != context_key
    }

    # convert df to have top 9 categories w.r.t. frequency of belonging to sequences + '(other)' for all other categories
    df = df.copy()
    for col in df.columns:
        if col == context_key:
            continue
        top_categories = sequences_per_category_dict[col].nlargest(9).index.tolist()
        not_in_top_categories_mask = ~df[col].isin(top_categories)
        if not_in_top_categories_mask.any():
            df[col] = df[col].cat.add_categories("(other)")
            df.loc[not_in_top_categories_mask, col] = "(other)"
            df[col] = df[col].cat.remove_unused_categories()
    sequences_per_category_binned_dict = {
        col: df.groupby(col)[context_key].nunique().rename_axis(None) for col in df.columns if col != context_key
    }

    # cnt_sum to be used for univariate plot normalization
    cnt_sum = df[context_key].nunique()

    # Example output for "team" (pd.Series):
    # team
    # ALT     18
    # ANA    164
    # Name: players_id, dtype: int64
    return sequences_per_category_dict, sequences_per_category_binned_dict, cnt_sum


def calculate_sequences_per_category_accuracy(
    *,
    seq_per_cat_trn_binned_cnts: pd.DataFrame,
    seq_per_cat_syn_binned_cnts: pd.DataFrame,
) -> pd.DataFrame:
    acc_seq_per_cat = pd.DataFrame(
        {"column": seq_per_cat_trn_binned_cnts.keys(), "accuracy": 0.5}
    )  # TODO: calculate seq_per_cat accuracy
    return acc_seq_per_cat


def plot_store_sequences_per_category(
    seq_per_cat_trn_cnts: dict[str, pd.Series],
    seq_per_cat_syn_cnts: dict[str, pd.Series],
    seq_per_cat_trn_binned_cnts: dict[str, pd.Series],
    seq_per_cat_syn_binned_cnts: dict[str, pd.Series],
    trn_cnt_sum: int,
    syn_cnt_sum: int,
    acc_seq_per_cat: pd.DataFrame,
    workspace: TemporaryWorkspace,
) -> None:
    with parallel_config("loky", n_jobs=min(cpu_count() - 1, 16)):
        Parallel()(
            delayed(plot_store_single_sequences_per_category)(
                row["column"],
                seq_per_cat_trn_cnts.get(row["column"]),
                seq_per_cat_syn_cnts.get(row["column"]),
                seq_per_cat_trn_binned_cnts.get(row["column"]),
                seq_per_cat_syn_binned_cnts.get(row["column"]),
                trn_cnt_sum,
                syn_cnt_sum,
                row["accuracy"],
                workspace,
            )
            for _, row in acc_seq_per_cat.iterrows()
        )


def plot_store_single_sequences_per_category(
    col: str,
    seq_per_cat_trn_cnts: pd.Series,
    seq_per_cat_syn_cnts: pd.Series,
    seq_per_cat_trn_binned_cnt: pd.Series,
    seq_per_cat_syn_binned_cnt: pd.Series,
    trn_cnt_sum: int,
    syn_cnt_sum: int,
    accuracy: float,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_univariate(
        col_name=col,
        trn_num_kde=None,
        syn_num_kde=None,
        trn_cat_col_cnts=seq_per_cat_trn_cnts,
        syn_cat_col_cnts=seq_per_cat_syn_cnts,
        trn_bin_col_cnts=seq_per_cat_trn_binned_cnt,
        syn_bin_col_cnts=seq_per_cat_syn_binned_cnt,
        accuracy=accuracy,
        trn_cnt_sum=trn_cnt_sum,
        syn_cnt_sum=syn_cnt_sum,
    )
    workspace.store_figure_html(fig, "sequences_per_category", col)
