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

import logging
from pathlib import Path

import pandas as pd

from mostlyai.qa import _accuracy, _html_report, _sampling
from mostlyai.qa._coherence import (
    calculate_distinct_categories_per_sequence,
    calculate_sequences_per_distinct_category,
    plot_store_distinct_categories_per_sequence,
    plot_store_sequences_per_distinct_category,
)
from mostlyai.qa._common import (
    REPORT_CREDITS,
    PrerequisiteNotMetError,
    ProgressCallback,
    ProgressCallbackWrapper,
    check_min_sample_size,
    check_statistics_prerequisite,
    determine_data_size,
)
from mostlyai.qa._filesystem import Statistics, TemporaryWorkspace
from mostlyai.qa._sampling import prepare_data_for_coherence

_LOG = logging.getLogger(__name__)


def report_from_statistics(
    *,
    syn_tgt_data: pd.DataFrame,
    syn_ctx_data: pd.DataFrame | None = None,
    statistics_path: str | Path | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    report_path: str | Path | None = "data-report.html",
    report_title: str = "Data Report",
    report_subtitle: str = "",
    report_credits: str = REPORT_CREDITS,
    max_sample_size_accuracy: int | None = None,
    max_sample_size_coherence: int | None = None,
    update_progress: ProgressCallback | None = None,
) -> Path:
    """
    Generate an HTML report based on previously generated statistics and newly provided synthetic data samples.

    Args:
        syn_tgt_data: The synthetic (target) data.
        syn_ctx_data: The synthetic context data.
        statistics_path: The path from where to fetch the statistics files.
        ctx_primary_key: The primary key of the context data.
        tgt_context_key: The context key of the target data.
        report_path: The path to store the HTML report.
        report_title: The title of the report.
        report_subtitle: The subtitle of the report.
        report_credits: The credits of the report.
        max_sample_size_accuracy: The maximum sample size for accuracy calculations.
        max_sample_size_coherence: The maximum sample size for coherence calculations.
        update_progress: The progress callback.

    Returns:
        The path to the generated HTML report.
    """

    with (
        TemporaryWorkspace() as workspace,
        ProgressCallbackWrapper(update_progress) as progress,
    ):
        # prepare report_path
        if report_path is None:
            report_path = Path.cwd() / "data-report.html"
        else:
            report_path = Path(report_path)

        statistics = Statistics(path=statistics_path)

        # determine sample size
        syn_sample_size = determine_data_size(syn_tgt_data, syn_ctx_data, ctx_primary_key, tgt_context_key)

        # early exit if prerequisites are not met
        try:
            check_statistics_prerequisite(statistics)
            check_min_sample_size(syn_sample_size, 100, "synthetic")
        except PrerequisiteNotMetError:
            _html_report.store_early_exit_report(report_path)
            return report_path

        meta = statistics.load_meta()

        # ensure synthetic data is structurally compatible with statistics
        if "trn_tgt_columns" in meta:
            syn_tgt_data = syn_tgt_data[meta["trn_tgt_columns"]]
        if "trn_ctx_columns" in meta and meta["trn_ctx_columns"] is not None:
            if syn_ctx_data is None:
                raise ValueError("syn_ctx_data is required for given statistics")
            syn_ctx_data = syn_ctx_data[meta["trn_ctx_columns"]]

        # prepare data
        _LOG.info("sample synthetic data started")
        syn = _sampling.prepare_data_for_accuracy(
            df_tgt=syn_tgt_data,
            df_ctx=syn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            # always pull Sequence Length and nxt columns for synthetic data
            # and let downstream functions decide if they are needed
            setup="1:N",
        )
        _LOG.info(f"sample synthetic data finished ({syn.shape=})")
        progress.update(completed=20, total=100)

        # calculate and plot accuracy and correlations
        acc_uni, acc_biv, acc_triv, corr_trn = _report_accuracy_and_correlations_from_statistics(
            syn=syn,
            statistics=statistics,
            workspace=workspace,
        )
        progress.update(completed=50, total=100)

        ori_coh_bins = statistics.load_coherence_bins()
        do_coherence = ori_coh_bins is not None
        if do_coherence:
            _LOG.info("prepare synthetic data for coherence started")
            syn_coh, _ = prepare_data_for_coherence(
                df_tgt=syn_tgt_data,
                tgt_context_key=tgt_context_key,
                bins=ori_coh_bins,
                max_sample_size=max_sample_size_coherence,
            )
            _LOG.info("report sequences per distinct category")
            acc_seqs_per_cat = _report_coherence_sequences_per_distinct_category(
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
            _LOG.info("report distinct categories per sequence")
            acc_cats_per_seq = _report_coherence_distinct_categories_per_sequence(
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
        else:
            acc_cats_per_seq = acc_seqs_per_cat = pd.DataFrame({"column": [], "accuracy": [], "accuracy_max": []})
        progress.update(completed=80, total=100)

        meta |= {
            "rows_synthetic": syn.shape[0],
            "report_title": report_title,
            "report_subtitle": report_subtitle,
            "report_credits": report_credits,
        }

        # HTML report
        _html_report.store_report(
            report_path=report_path,
            report_type="data_report",
            workspace=workspace,
            metrics=None,
            meta=meta,
            acc_uni=acc_uni,
            acc_biv=acc_biv,
            acc_triv=acc_triv,
            corr_trn=corr_trn,
            acc_cats_per_seq=acc_cats_per_seq,
            acc_seqs_per_cat=acc_seqs_per_cat,
        )
        progress.update(completed=100, total=100)
        _LOG.info(f"report stored at {report_path}")
        return report_path


def _report_accuracy_and_correlations_from_statistics(
    *,
    syn: pd.DataFrame,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _LOG.info("load original bins")
    bins = statistics.load_bins()
    syn = syn[bins.keys()].copy()

    _LOG.info("calculate synthetic bins")
    syn_bin, _ = _accuracy.bin_data(syn, bins)

    _LOG.info("load univariates, bivariates, trivariates")
    acc_uni = statistics.load_univariate_accuracies()
    acc_biv = statistics.load_bivariate_accuracies()
    acc_triv = statistics.load_trivariate_accuracies()

    _LOG.info("load numeric KDEs")
    ori_num_kdes = statistics.load_numeric_uni_kdes()

    _LOG.info("load categorical counts")
    ori_cat_uni_cnts = statistics.load_categorical_uni_counts()

    _LOG.info("load bin counts")
    ori_bin_cnts_uni, ori_bin_cnts_biv = statistics.load_bin_counts()

    _LOG.info("load correlations")
    corr_ori = statistics.load_correlations()

    _LOG.info("calculate synthetic correlations")
    corr_syn = _accuracy.calculate_correlations(binned=syn_bin, corr_cols=corr_ori.columns)

    _LOG.info("plot correlations")
    _accuracy.plot_store_correlation_matrices(corr_ori=corr_ori, corr_syn=corr_syn, workspace=workspace)

    _LOG.info("filter columns for plotting")
    syn = syn[acc_uni["column"]]
    acc_cols = list(set(acc_uni["column"]) | set(acc_biv["col1"]) | set(acc_biv["col2"]))
    syn_bin = syn_bin[acc_cols]

    _LOG.info("calculate numeric KDEs for synthetic")
    syn_num_kdes = _accuracy.calculate_numeric_uni_kdes(df=syn, ori_kdes=ori_num_kdes)

    _LOG.info("calculate categorical counts for synthetic")
    syn_cat_uni_cnts = _accuracy.calculate_categorical_uni_counts(
        df=syn,
        ori_col_counts=ori_cat_uni_cnts,
        hash_rare_values=False,
    )

    _LOG.info("calculate bin counts for synthetic")
    syn_bin_cnts_uni, syn_bin_cnts_biv = _accuracy.calculate_bin_counts(syn_bin)

    _LOG.info("plot univariates")
    _accuracy.plot_store_univariates(
        ori_num_kdes=ori_num_kdes,
        syn_num_kdes=syn_num_kdes,
        ori_cat_cnts=ori_cat_uni_cnts,
        syn_cat_cnts=syn_cat_uni_cnts,
        ori_cnts_uni=ori_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        acc_uni=acc_uni,
        workspace=workspace,
        show_accuracy=False,
    )

    _LOG.info("plot bivariates")
    _accuracy.plot_store_bivariates(
        ori_cnts_uni=ori_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        ori_cnts_biv=ori_bin_cnts_biv,
        syn_cnts_biv=syn_bin_cnts_biv,
        acc_biv=acc_biv,
        workspace=workspace,
        show_accuracy=False,
    )

    return acc_uni, acc_biv, acc_triv, corr_ori


def _report_coherence_distinct_categories_per_sequence(
    *,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    # calculate distinct categories per sequence
    _LOG.info("calculate distinct categories per sequence for synthetic")
    syn_cats_per_seq = calculate_distinct_categories_per_sequence(df=syn_coh, context_key=tgt_context_key)

    # bin distinct categories per sequence
    _LOG.info("load distinct categories per sequence bins for training")
    bins = statistics.load_distinct_categories_per_sequence_bins()
    _LOG.info("bin distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq, _ = _accuracy.bin_data(syn_cats_per_seq, bins=bins)

    # prepare KDEs for distribution (left) plots
    _LOG.info("load KDEs of distinct categories per sequence for training")
    ori_cats_per_seq_kdes = statistics.load_distinct_categories_per_sequence_kdes()
    _LOG.info("calculate KDEs of distinct categories per sequence for synthetic")
    syn_cats_per_seq_kdes = _accuracy.calculate_numeric_uni_kdes(df=syn_cats_per_seq, ori_kdes=ori_cats_per_seq_kdes)

    # prepare counts for binned (right) plots
    _LOG.info("load counts of binned distinct categories per sequence for training")
    ori_binned_cats_per_seq_cnts = statistics.load_binned_distinct_categories_per_sequence_counts()
    _LOG.info("calculate counts of binned distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq_cnts = _accuracy.calculate_categorical_uni_counts(
        df=syn_binned_cats_per_seq, hash_rare_values=False
    )

    # load per-column accuracy
    _LOG.info("load distinct categories per sequence accuracy")
    acc_cats_per_seq = statistics.load_distinct_categories_per_sequence_accuracy()

    # make plots
    _LOG.info("plot and store distinct categories per sequence")
    plot_store_distinct_categories_per_sequence(
        ori_cats_per_seq_kdes=ori_cats_per_seq_kdes,
        syn_cats_per_seq_kdes=syn_cats_per_seq_kdes,
        ori_binned_cats_per_seq_cnts=ori_binned_cats_per_seq_cnts,
        syn_binned_cats_per_seq_cnts=syn_binned_cats_per_seq_cnts,
        acc_cats_per_seq=acc_cats_per_seq,
        workspace=workspace,
    )
    return acc_cats_per_seq


def _report_coherence_sequences_per_distinct_category(
    *,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    _LOG.info("load sequences per distinct category artifacts for training")
    seqs_per_cat_cnts, seqs_per_top_cat_cnts, top_cats, ori_n_seqs = (
        statistics.load_sequences_per_distinct_category_artifacts()
    )

    _LOG.info("calculate sequences per distinct category for synthetic")
    seqs_per_cat_syn_cnts, seqs_per_cat_syn_binned_cnts, _, syn_cnt_sum = calculate_sequences_per_distinct_category(
        df=syn_coh,
        context_key=tgt_context_key,
        top_cats=top_cats,
    )

    # load per-column accuracy
    _LOG.info("load sequences per distinct category accuracy")
    acc_seqs_per_cat = statistics.load_sequences_per_distinct_category_accuracy()

    # make plots
    _LOG.info("plot and store sequences per distinct category")
    plot_store_sequences_per_distinct_category(
        ori_seqs_per_cat_cnts=seqs_per_cat_cnts,
        syn_seqs_per_cat_cnts=seqs_per_cat_syn_cnts,
        ori_seqs_per_top_cat_cnts=seqs_per_top_cat_cnts,
        syn_seqs_per_top_cat_cnts=seqs_per_cat_syn_binned_cnts,
        ori_n_seqs=ori_n_seqs,
        syn_n_seqs=syn_cnt_sum,
        acc_seqs_per_cat=acc_seqs_per_cat,
        workspace=workspace,
    )

    return acc_seqs_per_cat
