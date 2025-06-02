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

import datetime
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from mostlyai.qa._accuracy import filter_biv_acc_for_plotting, filter_uni_acc_for_plotting, trim_label
from mostlyai.qa._common import TGT_COLUMN_PREFIX
from mostlyai.qa._filesystem import TemporaryWorkspace
from mostlyai.qa.assets import (
    HTML_ASSETS_PATH,
    HTML_REPORT_EARLY_EXIT,
    HTML_REPORT_TEMPLATE,
    read_html_assets,
)
from mostlyai.qa.metrics import ModelMetrics

_LOG = logging.getLogger(__name__)


def get_uni_htmls(acc_uni: pd.DataFrame, workspace: TemporaryWorkspace) -> list[str]:
    paths_uni = workspace.get_figure_paths("univariate", acc_uni[["column"]]).values()
    return [f.read_text(encoding="utf-8") for f in paths_uni]


def get_cats_per_seq_htmls(acc_cats_per_seq: pd.DataFrame, workspace: TemporaryWorkspace) -> list[str]:
    paths_cats_per_seq = workspace.get_figure_paths(
        "distinct_categories_per_sequence", acc_cats_per_seq[["column"]]
    ).values()
    return [f.read_text(encoding="utf-8") for f in paths_cats_per_seq]


def get_seqs_per_cat_htmls(acc_seqs_per_cat: pd.DataFrame, workspace: TemporaryWorkspace) -> list[str]:
    paths_seqs_per_cat = workspace.get_figure_paths(
        "sequences_per_distinct_category", acc_seqs_per_cat[["column"]]
    ).values()
    return [f.read_text(encoding="utf-8") for f in paths_seqs_per_cat]


def get_biv_htmls(acc_biv: pd.DataFrame, workspace: TemporaryWorkspace) -> tuple[list[str], list[str], list[str]]:
    acc_biv_ctx = acc_biv.loc[acc_biv.type == "ctx"]
    acc_biv_tgt = acc_biv.loc[acc_biv.type == "tgt"]
    acc_biv_nxt = acc_biv.loc[acc_biv.type == "nxt"]
    paths_biv_ctx = workspace.get_figure_paths("bivariate", acc_biv_ctx[["col1", "col2"]]).values()
    paths_biv_tgt = workspace.get_figure_paths("bivariate", acc_biv_tgt[["col1", "col2"]]).values()
    paths_biv_nxt = workspace.get_figure_paths("bivariate", acc_biv_nxt[["col1", "col2"]]).values()
    html_biv_ctx = [f.read_text(encoding="utf-8") for f in paths_biv_ctx]
    html_biv_tgt = [f.read_text(encoding="utf-8") for f in paths_biv_tgt]
    html_biv_nxt = [f.read_text(encoding="utf-8") for f in paths_biv_nxt]
    return html_biv_ctx, html_biv_tgt, html_biv_nxt


def store_report(
    report_path: Path,
    report_type: Literal["model_report", "data_report"],
    workspace: TemporaryWorkspace,
    metrics: ModelMetrics | None,
    meta: dict,
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    acc_triv: pd.DataFrame,
    acc_cats_per_seq: pd.DataFrame,
    acc_seqs_per_cat: pd.DataFrame,
    corr_trn: pd.DataFrame,
):
    """
    Render HTML report.
    """

    # summarize accuracies by column for overview table
    accuracy_table_by_column = summarize_accuracies_by_column(
        acc_uni, acc_biv, acc_triv, acc_cats_per_seq, acc_seqs_per_cat
    )
    accuracy_table_by_column = accuracy_table_by_column.sort_values("univariate", ascending=False)

    acc_uni = filter_uni_acc_for_plotting(acc_uni)
    html_uni = get_uni_htmls(acc_uni=acc_uni, workspace=workspace)
    html_cats_per_seq = get_cats_per_seq_htmls(acc_cats_per_seq=acc_cats_per_seq, workspace=workspace)
    html_seqs_per_cat = get_seqs_per_cat_htmls(acc_seqs_per_cat=acc_seqs_per_cat, workspace=workspace)
    acc_biv = filter_biv_acc_for_plotting(acc_biv, corr_trn)
    html_biv_ctx, html_biv_tgt, html_biv_nxt = get_biv_htmls(acc_biv=acc_biv, workspace=workspace)

    correlation_matrix_html_chart = workspace.get_unique_figure_path("correlation_matrices").read_text(encoding="utf-8")
    similarity_pca_html_chart_path = workspace.get_unique_figure_path("similarity_pca")
    similarity_pca_html_chart = None
    if similarity_pca_html_chart_path.exists():
        similarity_pca_html_chart = similarity_pca_html_chart_path.read_text(encoding="utf-8")
    if report_type == "model_report":
        accuracy_matrix_html_chart = workspace.get_unique_figure_path("accuracy_matrix").read_text(encoding="utf-8")
        distances_dcr_html_chart = workspace.get_unique_figure_path("distances_dcr").read_text(encoding="utf-8")
    else:
        accuracy_matrix_html_chart = None
        distances_dcr_html_chart = None

    meta |= {
        "report_creation_datetime": datetime.datetime.now(),
    }

    template = Environment(loader=FileSystemLoader(HTML_ASSETS_PATH)).get_template(HTML_REPORT_TEMPLATE)
    html = template.render(
        is_model_report=(report_type == "model_report"),
        html_assets=read_html_assets(),
        report_creation_datetime=datetime.datetime.now(),
        metrics=metrics.model_dump() if metrics else None,
        meta=meta,
        accuracy_table_by_column=accuracy_table_by_column,
        accuracy_matrix_html_chart=accuracy_matrix_html_chart,
        correlation_matrix_html_chart=correlation_matrix_html_chart,
        similarity_pca_html_chart=similarity_pca_html_chart,
        distances_dcr_html_chart=distances_dcr_html_chart,
        univariate_html_charts=html_uni,
        distinct_categories_per_sequence_html_charts=html_cats_per_seq,
        sequences_per_distinct_category_html_charts=html_seqs_per_cat,
        bivariate_html_charts_tgt=html_biv_tgt,
        bivariate_html_charts_ctx=html_biv_ctx,
        bivariate_html_charts_nxt=html_biv_nxt,
    )
    report_path.write_text(html, encoding="utf-8")


def summarize_accuracies_by_column(
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    acc_triv: pd.DataFrame,
    acc_cats_per_seq: pd.DataFrame,
    acc_seqs_per_cat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates DataFrame that stores per-column univariate, bivariate, trivariate and coherence accuracies.
    """

    tbl_acc_uni = acc_uni.rename(columns={"accuracy": "univariate", "accuracy_max": "univariate_max"})
    tbl_acc = tbl_acc_uni

    tbl_acc_biv = (
        acc_biv.melt(value_vars=["col1", "col2"], value_name="column", id_vars=["accuracy", "accuracy_max"])
        .groupby("column")[["accuracy", "accuracy_max"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "bivariate",
                "accuracy_max": "bivariate_max",
            }
        )
    )
    if not tbl_acc_biv.empty:
        tbl_acc = tbl_acc_uni.merge(tbl_acc_biv, how="left")

    tbl_acc_triv = (
        acc_triv.melt(value_vars=["col1", "col2", "col3"], value_name="column", id_vars=["accuracy", "accuracy_max"])
        .groupby("column")[["accuracy", "accuracy_max"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "trivariate",
                "accuracy_max": "trivariate_max",
            }
        )
    )
    if not tbl_acc_triv.empty:
        tbl_acc = tbl_acc.merge(tbl_acc_triv, how="left")

    acc_nxt = acc_biv.loc[acc_biv.type == "nxt"]
    if not all((acc_nxt.empty, acc_cats_per_seq.empty, acc_seqs_per_cat.empty)):
        acc_nxt = acc_nxt.groupby("col1").mean(["accuracy", "accuracy_max"]).reset_index(names="column")
        acc_nxt = acc_nxt[acc_nxt["column"].str.startswith(TGT_COLUMN_PREFIX)]
        acc_cats_per_seq = acc_cats_per_seq.assign(column=TGT_COLUMN_PREFIX + acc_cats_per_seq["column"])
        acc_seqs_per_cat = acc_seqs_per_cat.assign(column=TGT_COLUMN_PREFIX + acc_seqs_per_cat["column"])
        tbl_acc_coherence = (
            pd.concat([a for a in [acc_nxt, acc_cats_per_seq, acc_seqs_per_cat] if not a.empty])
            .groupby("column")
            .mean(["accuracy", "accuracy_max"])
            .rename(columns={"accuracy": "coherence", "accuracy_max": "coherence_max"})
            .reset_index()
        )
        tbl_acc = tbl_acc.merge(tbl_acc_coherence, how="left")

    tbl_acc["column"] = tbl_acc["column"].apply(lambda y: trim_label(y))
    return tbl_acc


def store_early_exit_report(report_path: Path):
    template = Environment(loader=FileSystemLoader(HTML_ASSETS_PATH)).get_template(HTML_REPORT_EARLY_EXIT)
    report_html = template.render(html_assets=read_html_assets(), meta={})
    report_path.write_text(report_html, encoding="utf-8")
