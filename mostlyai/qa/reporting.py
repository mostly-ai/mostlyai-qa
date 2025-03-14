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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_dtype

from mostlyai.qa import _distances, _similarity, _html_report
from mostlyai.qa._accuracy import (
    bin_data,
    binning_data,
    calculate_correlations,
    plot_store_correlation_matrices,
    calculate_univariates,
    calculate_bivariates,
    plot_store_accuracy_matrix,
    filter_uni_acc_for_plotting,
    filter_biv_acc_for_plotting,
    calculate_numeric_uni_kdes,
    calculate_categorical_uni_counts,
    calculate_bin_counts,
    plot_store_univariates,
    plot_store_bivariates,
)
from mostlyai.qa._coherence import (
    calculate_distinct_categories_per_sequence,
    calculate_distinct_categories_per_sequence_accuracy,
    calculate_sequences_per_distinct_category,
    calculate_sequences_per_distinct_category_accuracy,
    plot_store_distinct_categories_per_sequence,
    plot_store_sequences_per_distinct_category,
)
from mostlyai.qa.metrics import ModelMetrics, Accuracy, Similarity, Distances
from mostlyai.qa._sampling import (
    calculate_embeddings,
    pull_data_for_accuracy,
    pull_data_for_coherence,
    pull_data_for_embeddings,
)
from mostlyai.qa._common import (
    determine_data_size,
    ProgressCallback,
    PrerequisiteNotMetError,
    check_min_sample_size,
    NXT_COLUMN,
    CTX_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
    REPORT_CREDITS,
    ProgressCallbackWrapper,
)
from mostlyai.qa._filesystem import Statistics, TemporaryWorkspace

_LOG = logging.getLogger(__name__)


def report(
    *,
    syn_tgt_data: pd.DataFrame,
    trn_tgt_data: pd.DataFrame,
    hol_tgt_data: pd.DataFrame | None = None,
    syn_ctx_data: pd.DataFrame | None = None,
    trn_ctx_data: pd.DataFrame | None = None,
    hol_ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    report_path: str | Path | None = "model-report.html",
    report_title: str = "Model Report",
    report_subtitle: str = "",
    report_credits: str = REPORT_CREDITS,
    max_sample_size_accuracy: int | None = None,
    max_sample_size_embeddings: int | None = None,
    max_sample_size_coherence: int | None = None,
    statistics_path: str | Path | None = None,
    update_progress: ProgressCallback | None = None,
) -> tuple[Path, ModelMetrics | None]:
    """
    Generate an HTML report and metrics for assessing synthetic data quality.

    Compares synthetic data samples with original training samples in terms of accuracy, similarity and distances.
    Provide holdout samples to calculate reference values for similarity and distances (recommended).

    If synthetic data has been generated conditionally on a context dataset, provide the context data as well. This
    will allow for bivariate accuracy metrics between context and target to be calculated.

    If the data represents sequential data, provide the `tgt_context_key` to set the groupby column for the target data.

    Customize the report with the `report_title`, `report_subtitle` and `report_credits`.

    Limit the compute time used by setting `max_sample_size_accuracy`, `max_sample_size_coherence` and `max_sample_size_embeddings`.

    Args:
        syn_tgt_data: The synthetic (target) data.
        trn_tgt_data: The training (target) data.
        hol_tgt_data: The holdout (target) data.
        syn_ctx_data: The synthetic context data.
        trn_ctx_data: The training context data.
        hol_ctx_data: The holdout context data.
        ctx_primary_key: The primary key of the context data.
        tgt_context_key: The context key of the target data.
        report_path: The path to store the HTML report.
        report_title: The title of the report.
        report_subtitle: The subtitle of the report.
        report_credits: The credits of the report.
        max_sample_size_accuracy: The maximum sample size for accuracy calculations.
        max_sample_size_coherence: The maximum sample size for coherence calculations.
        max_sample_size_embeddings: The maximum sample size for embedding calculations (similarity & distances)
        statistics_path: The path of where to store the statistics to be used by `report_from_statistics`
        update_progress: The progress callback.

    Returns:
        The path to the generated HTML report.
        Metrics instance with accuracy, similarity, and distances metrics.
    """

    if syn_ctx_data is not None:
        if ctx_primary_key is None:
            raise ValueError("If syn_ctx_data is provided, then ctx_primary_key must also be provided.")
        if trn_ctx_data is None:
            raise ValueError("If syn_ctx_data is provided, then trn_ctx_data must also be provided.")
        if hol_tgt_data is not None and hol_ctx_data is None:
            raise ValueError("If syn_ctx_data is provided, then hol_ctx_data must also be provided.")

    with (
        TemporaryWorkspace() as workspace,
        ProgressCallbackWrapper(update_progress) as progress,
    ):
        # ensure all columns are present and in the same order as training data
        syn_tgt_data = syn_tgt_data[trn_tgt_data.columns]
        if hol_tgt_data is not None:
            hol_tgt_data = hol_tgt_data[trn_tgt_data.columns]
        if syn_ctx_data is not None and trn_ctx_data is not None:
            syn_ctx_data = syn_ctx_data[trn_ctx_data.columns]
        if hol_ctx_data is not None and trn_ctx_data is not None:
            hol_ctx_data = hol_ctx_data[trn_ctx_data.columns]

        # prepare report_path
        if report_path is None:
            report_path = Path.cwd() / "model-report.html"
        else:
            report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # prepare statistics_path
        if statistics_path is None:
            statistics_path = Path(workspace.name) / "statistics"
        else:
            statistics_path = Path(statistics_path)
        statistics_path.mkdir(parents=True, exist_ok=True)
        statistics = Statistics(path=statistics_path)

        # determine sample sizes
        syn_sample_size = determine_data_size(syn_tgt_data, syn_ctx_data, ctx_primary_key, tgt_context_key)
        trn_sample_size = determine_data_size(trn_tgt_data, trn_ctx_data, ctx_primary_key, tgt_context_key)
        if hol_tgt_data is not None:
            hol_sample_size = determine_data_size(hol_tgt_data, hol_ctx_data, ctx_primary_key, tgt_context_key)
        else:
            hol_sample_size = 0

        # early exit if prerequisites are not met
        try:
            check_min_sample_size(syn_sample_size, 100, "synthetic")
            check_min_sample_size(trn_sample_size, 90, "training")
            if hol_tgt_data is not None:
                check_min_sample_size(hol_sample_size, 10, "holdout")
        except PrerequisiteNotMetError as err:
            _LOG.info(err)
            statistics.mark_early_exit()
            _html_report.store_early_exit_report(report_path)
            return report_path, None

        # prepare datasets for accuracy
        if trn_ctx_data is not None:
            assert ctx_primary_key is not None
            setup = (
                "1:1"
                if (
                    trn_ctx_data[ctx_primary_key].is_unique
                    and trn_tgt_data[tgt_context_key].is_unique
                    and set(trn_ctx_data[ctx_primary_key]) == set(trn_tgt_data[tgt_context_key])
                )
                else "1:N"
            )
        elif tgt_context_key is not None:
            setup = "1:1" if trn_tgt_data[tgt_context_key].is_unique else "1:N"
        else:
            setup = "1:1"

        _LOG.info("prepare synthetic data for accuracy started")
        syn = pull_data_for_accuracy(
            df_tgt=syn_tgt_data,
            df_ctx=syn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            setup=setup,
        )
        progress.update(completed=5, total=100)

        _LOG.info("prepare training data for accuracy started")
        trn = pull_data_for_accuracy(
            df_tgt=trn_tgt_data,
            df_ctx=trn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            setup=setup,
        )
        progress.update(completed=10, total=100)

        # coerce dtypes to match the original training data dtypes
        for col in trn:
            if is_numeric_dtype(trn[col]):
                syn[col] = pd.to_numeric(syn[col], errors="coerce")
            elif is_datetime64_dtype(trn[col]):
                syn[col] = pd.to_datetime(syn[col], errors="coerce")
            syn[col] = syn[col].astype(trn[col].dtype)

        _LOG.info("report accuracy and correlations")
        acc_uni, acc_biv, corr_trn = _report_accuracy_and_correlations(
            trn=trn,
            syn=syn,
            statistics=statistics,
            workspace=workspace,
        )
        progress.update(completed=20, total=100)

        # ensure that embeddings are all equal size for a fair 3-way comparison
        max_sample_size_embeddings_final = min(
            max_sample_size_embeddings or float("inf"),
            syn_sample_size,
            trn_sample_size,
            hol_sample_size or float("inf"),
        )

        if max_sample_size_embeddings_final > 50_000 and max_sample_size_embeddings is None:
            warnings.warn(
                UserWarning(
                    "More than 50k embeddings will be calculated per dataset, which may take a long time. "
                    "Consider setting a limit via `max_sample_size_embeddings` to speed up the process."
                )
            )

        do_coherence = setup == "1:N"
        if do_coherence:
            _LOG.info("prepare training data for coherence started")
            trn_coh, trn_coh_bins = pull_data_for_coherence(
                df_tgt=trn_tgt_data, tgt_context_key=tgt_context_key, max_sample_size=max_sample_size_coherence
            )
            _LOG.info("prepare synthetic data for coherence started")
            syn_coh, _ = pull_data_for_coherence(
                df_tgt=syn_tgt_data,
                tgt_context_key=tgt_context_key,
                bins=trn_coh_bins,
                max_sample_size=max_sample_size_coherence,
            )
            _LOG.info("store bins used for training data for coherence")
            statistics.store_coherence_bins(bins=trn_coh_bins)
            _LOG.info("report sequences per distinct category")
            acc_seqs_per_cat = _report_coherence_sequences_per_distinct_category(
                trn_coh=trn_coh,
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
            _LOG.info("report distinct categories per sequence")
            acc_cats_per_seq = _report_coherence_distinct_categories_per_sequence(
                trn_coh=trn_coh,
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
        else:
            acc_cats_per_seq = acc_seqs_per_cat = pd.DataFrame({"column": [], "accuracy": [], "accuracy_max": []})
        progress.update(completed=25, total=100)

        _LOG.info("calculate embeddings for synthetic")
        syn_embeds = calculate_embeddings(
            strings=pull_data_for_embeddings(
                df_tgt=syn_tgt_data,
                df_ctx=syn_ctx_data,
                ctx_primary_key=ctx_primary_key,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_embeddings_final,
            ),
            progress=progress,
            progress_from=25,
            progress_to=45,
        )
        _LOG.info("calculate embeddings for training")
        trn_embeds = calculate_embeddings(
            strings=pull_data_for_embeddings(
                df_tgt=trn_tgt_data,
                df_ctx=trn_ctx_data,
                ctx_primary_key=ctx_primary_key,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_embeddings_final,
            ),
            progress=progress,
            progress_from=45,
            progress_to=65,
        )
        if hol_tgt_data is not None:
            _LOG.info("calculate embeddings for holdout")
            hol_embeds = calculate_embeddings(
                strings=pull_data_for_embeddings(
                    df_tgt=hol_tgt_data,
                    df_ctx=hol_ctx_data,
                    ctx_primary_key=ctx_primary_key,
                    tgt_context_key=tgt_context_key,
                    max_sample_size=max_sample_size_embeddings_final,
                ),
                progress=progress,
                progress_from=65,
                progress_to=85,
            )
        else:
            hol_embeds = None
        progress.update(completed=85, total=100)

        _LOG.info("report similarity")
        sim_cosine_trn_hol, sim_cosine_trn_syn, sim_auc_trn_hol, sim_auc_trn_syn = _report_similarity(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds,
            workspace=workspace,
            statistics=statistics,
        )
        progress.update(completed=95, total=100)

        _LOG.info("report distances")
        dcr_syn_trn, dcr_syn_hol, dcr_trn_hol = _report_distances(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds,
            workspace=workspace,
        )
        progress.update(completed=99, total=100)

        metrics = _calculate_metrics(
            acc_uni=acc_uni,
            acc_biv=acc_biv,
            dcr_syn_trn=dcr_syn_trn,
            dcr_syn_hol=dcr_syn_hol,
            dcr_trn_hol=dcr_trn_hol,
            sim_cosine_trn_hol=sim_cosine_trn_hol,
            sim_cosine_trn_syn=sim_cosine_trn_syn,
            sim_auc_trn_hol=sim_auc_trn_hol,
            sim_auc_trn_syn=sim_auc_trn_syn,
            acc_cats_per_seq=acc_cats_per_seq,
            acc_seqs_per_cat=acc_seqs_per_cat,
        )
        meta = {
            "rows_original": trn_sample_size + hol_sample_size,
            "rows_training": trn_sample_size,
            "rows_holdout": hol_sample_size,
            "rows_synthetic": syn_sample_size,
            "tgt_columns": len([c for c in trn.columns if c.startswith(TGT_COLUMN_PREFIX)]),
            "ctx_columns": len([c for c in trn.columns if c.startswith(CTX_COLUMN_PREFIX)]),
            "trn_tgt_columns": trn_tgt_data.columns.to_list(),
            "trn_ctx_columns": trn_ctx_data.columns.to_list() if trn_ctx_data is not None else None,
            "report_title": report_title,
            "report_subtitle": report_subtitle,
            "report_credits": report_credits,
        }
        statistics.store_meta(meta=meta)
        _html_report.store_report(
            report_path=report_path,
            report_type="model_report",
            workspace=workspace,
            metrics=metrics,
            meta=meta,
            acc_uni=acc_uni,
            acc_cats_per_seq=acc_cats_per_seq,
            acc_seqs_per_cat=acc_seqs_per_cat,
            acc_biv=acc_biv,
            corr_trn=corr_trn,
        )
        progress.update(completed=100, total=100)
        return report_path, metrics


def _calculate_metrics(
    *,
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    dcr_syn_trn: np.ndarray,
    dcr_syn_hol: np.ndarray | None,
    dcr_trn_hol: np.ndarray | None = None,
    sim_cosine_trn_hol: np.float64,
    sim_cosine_trn_syn: np.float64,
    sim_auc_trn_hol: np.float64,
    sim_auc_trn_syn: np.float64,
    acc_cats_per_seq: pd.DataFrame,
    acc_seqs_per_cat: pd.DataFrame,
) -> ModelMetrics:
    # univariates
    acc_univariate = acc_uni.accuracy.mean()
    acc_univariate_max = acc_uni.accuracy_max.mean()
    # bivariates
    acc_tgt_ctx = acc_biv.loc[acc_biv.type != NXT_COLUMN]
    if not acc_tgt_ctx.empty:
        acc_bivariate = acc_tgt_ctx.accuracy.mean()
        acc_bivariate_max = acc_tgt_ctx.accuracy_max.mean()
    else:
        acc_bivariate = acc_bivariate_max = None
    # coherence
    acc_nxt = acc_biv.loc[acc_biv.type == NXT_COLUMN]
    nxt_col_coherence = nxt_col_coherence_max = None
    if not acc_nxt.empty:
        nxt_col_coherence = acc_nxt.accuracy.mean()
        nxt_col_coherence_max = acc_nxt.accuracy_max.mean()
    cats_per_seq_coherence = cats_per_seq_coherence_max = None
    if not acc_cats_per_seq.empty:
        cats_per_seq_coherence = acc_cats_per_seq.accuracy.mean()
        cats_per_seq_coherence_max = acc_cats_per_seq.accuracy_max.mean()
    seqs_per_cat_coherence = seqs_per_cat_coherence_max = None
    if not acc_seqs_per_cat.empty:
        seqs_per_cat_coherence = acc_seqs_per_cat.accuracy.mean()
        seqs_per_cat_coherence_max = acc_seqs_per_cat.accuracy_max.mean()
    coherence_metrics = [
        m for m in (nxt_col_coherence, cats_per_seq_coherence, seqs_per_cat_coherence) if m is not None
    ]
    coherence_max_metrics = [
        m for m in (nxt_col_coherence_max, cats_per_seq_coherence_max, seqs_per_cat_coherence_max) if m is not None
    ]
    acc_coherence = np.mean(coherence_metrics) if coherence_metrics else None
    acc_coherence_max = np.mean(coherence_max_metrics) if coherence_max_metrics else None
    # calculate overall accuracy
    acc_overall = np.mean([m for m in (acc_univariate, acc_bivariate, acc_coherence) if m is not None])
    acc_overall_max = np.mean([m for m in (acc_univariate_max, acc_bivariate_max, acc_coherence_max) if m is not None])
    accuracy = Accuracy(
        overall=acc_overall,
        univariate=acc_univariate,
        bivariate=acc_bivariate,
        coherence=acc_coherence,
        overall_max=acc_overall_max,
        univariate_max=acc_univariate_max,
        bivariate_max=acc_bivariate_max,
        coherence_max=acc_coherence_max,
    )
    similarity = Similarity(
        cosine_similarity_training_synthetic=sim_cosine_trn_syn,
        cosine_similarity_training_holdout=sim_cosine_trn_hol if sim_cosine_trn_hol is not None else None,
        discriminator_auc_training_synthetic=sim_auc_trn_syn,
        discriminator_auc_training_holdout=sim_auc_trn_hol if sim_auc_trn_hol is not None else None,
    )
    distances = Distances(
        ims_training=(dcr_syn_trn <= 1e-6).mean(),
        ims_holdout=(dcr_syn_hol <= 1e-6).mean() if dcr_syn_hol is not None else None,
        ims_trn_hol=(dcr_trn_hol <= 1e-6).mean() if dcr_trn_hol is not None else None,
        dcr_training=dcr_syn_trn.mean(),
        dcr_holdout=dcr_syn_hol.mean() if dcr_syn_hol is not None else None,
        dcr_trn_hol=dcr_trn_hol.mean() if dcr_trn_hol is not None else None,
        dcr_share=np.mean(dcr_syn_trn < dcr_syn_hol) + np.mean(dcr_syn_trn == dcr_syn_hol) / 2
        if dcr_syn_hol is not None
        else None,
    )
    return ModelMetrics(
        accuracy=accuracy,
        similarity=similarity,
        distances=distances,
    )


def _report_accuracy_and_correlations(
    *,
    trn: pd.DataFrame,
    syn: pd.DataFrame,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # bin data
    trn_bin, syn_bin = binning_data(
        trn=trn,
        syn=syn,
        statistics=statistics,
    )

    # calculate correlations for original data
    trn_corr = calculate_correlations(binned=trn_bin)

    # store correlations for original data
    statistics.store_correlations(trn_corr=trn_corr)

    # calculate correlations for synthetic data
    corr_syn = calculate_correlations(binned=syn_bin, corr_cols=trn_corr.columns)

    # plot correlations matrices
    plot_store_correlation_matrices(corr_trn=trn_corr, corr_syn=corr_syn, workspace=workspace)

    # calculate univariate accuracies
    acc_uni = calculate_univariates(trn_bin, syn_bin)

    # calculate bivariate accuracies
    acc_biv = calculate_bivariates(trn_bin, syn_bin)

    # plot and store accuracy matrix
    plot_store_accuracy_matrix(
        acc_uni=acc_uni,
        acc_biv=acc_biv,
        workspace=workspace,
    )

    # filter columns for plotting
    acc_uni_plt = filter_uni_acc_for_plotting(acc_uni)
    acc_biv_plt = filter_biv_acc_for_plotting(acc_biv, trn_corr)
    trn = trn[acc_uni_plt["column"]]
    syn = syn[acc_uni_plt["column"]]
    acc_cols_plt = list(set(acc_uni["column"]) | set(acc_biv["col1"]) | set(acc_biv["col2"]))
    trn_bin = trn_bin[acc_cols_plt]
    syn_bin = syn_bin[acc_cols_plt]

    # store univariate and bivariate accuracies
    statistics.store_univariate_accuracies(acc_uni)
    statistics.store_bivariate_accuracies(acc_biv)

    # calculate KDEs for original
    trn_num_kdes = calculate_numeric_uni_kdes(trn)

    # store KDEs for original
    statistics.store_numeric_uni_kdes(trn_num_kdes)

    # calculate KDEs for synthetic
    syn_num_kdes = calculate_numeric_uni_kdes(syn, trn_num_kdes)

    # calculate categorical counts for original
    trn_cat_uni_cnts = calculate_categorical_uni_counts(df=trn, hash_rare_values=True)

    # store categorical counts for original
    statistics.store_categorical_uni_counts(trn_cat_uni_cnts)

    # calculate categorical counts for synthetic
    syn_cat_uni_cnts = calculate_categorical_uni_counts(
        df=syn,
        trn_col_counts=trn_cat_uni_cnts,
        hash_rare_values=False,
    )

    # calculate bin counts for original
    trn_bin_cnts_uni, trn_bin_cnts_biv = calculate_bin_counts(trn_bin)

    # store bin counts for original
    statistics.store_bin_counts(trn_cnts_uni=trn_bin_cnts_uni, trn_cnts_biv=trn_bin_cnts_biv)

    # calculate bin counts for synthetic
    syn_bin_cnts_uni, syn_bin_cnts_biv = calculate_bin_counts(binned=syn_bin)

    # plot univariate distributions
    plot_store_univariates(
        trn_num_kdes=trn_num_kdes,
        syn_num_kdes=syn_num_kdes,
        trn_cat_cnts=trn_cat_uni_cnts,
        syn_cat_cnts=syn_cat_uni_cnts,
        trn_cnts_uni=trn_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        acc_uni=acc_uni_plt,
        workspace=workspace,
        show_accuracy=True,
    )

    # plot bivariate distributions
    plot_store_bivariates(
        trn_cnts_uni=trn_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        trn_cnts_biv=trn_bin_cnts_biv,
        syn_cnts_biv=syn_bin_cnts_biv,
        acc_biv=acc_biv_plt,
        workspace=workspace,
        show_accuracy=True,
    )

    return acc_uni, acc_biv, trn_corr


def _report_coherence_distinct_categories_per_sequence(
    *,
    trn_coh: pd.DataFrame,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    # calculate distinct categories per sequence
    _LOG.info("calculate distinct categories per sequence for training")
    trn_cats_per_seq = calculate_distinct_categories_per_sequence(df=trn_coh, context_key=tgt_context_key)
    _LOG.info("calculate distinct categories per sequence for synthetic")
    syn_cats_per_seq = calculate_distinct_categories_per_sequence(df=syn_coh, context_key=tgt_context_key)

    # bin distinct categories per sequence
    _LOG.info("bin distinct categories per sequence for training")
    trn_binned_cats_per_seq, bins = bin_data(trn_cats_per_seq, bins=10)
    _LOG.info("store distinct categories per sequence bins for training")
    statistics.store_distinct_categories_per_sequence_bins(bins=bins)
    _LOG.info("bin distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq, _ = bin_data(syn_cats_per_seq, bins=bins)

    # prepare KDEs for distribution (left) plots
    _LOG.info("calculate KDEs of distinct categories per sequence for training")
    trn_cats_per_seq_kdes = calculate_numeric_uni_kdes(df=trn_cats_per_seq)
    _LOG.info("store KDEs of distinct categories per sequence for training")
    statistics.store_distinct_categories_per_sequence_kdes(trn_kdes=trn_cats_per_seq_kdes)
    _LOG.info("calculate KDEs of distinct categories per sequence for synthetic")
    syn_cats_per_seq_kdes = calculate_numeric_uni_kdes(df=syn_cats_per_seq, trn_kdes=trn_cats_per_seq_kdes)

    # prepare counts for binned (right) plots
    _LOG.info("calculate counts of binned distinct categories per sequence for training")
    trn_binned_cats_per_seq_cnts = calculate_categorical_uni_counts(df=trn_binned_cats_per_seq, hash_rare_values=False)
    _LOG.info("store counts of binned distinct categories per sequence for training")
    statistics.store_binned_distinct_categories_per_sequence_counts(counts=trn_binned_cats_per_seq_cnts)
    _LOG.info("calculate counts of binned distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq_cnts = calculate_categorical_uni_counts(df=syn_binned_cats_per_seq, hash_rare_values=False)

    # calculate per-column accuracy
    _LOG.info("calculate distinct categories per sequence accuracy")
    acc_cats_per_seq = calculate_distinct_categories_per_sequence_accuracy(
        trn_binned_cats_per_seq=trn_binned_cats_per_seq, syn_binned_cats_per_seq=syn_binned_cats_per_seq
    )
    _LOG.info("store distinct categories per sequence accuracy")
    statistics.store_distinct_categories_per_sequence_accuracy(accuracy=acc_cats_per_seq)

    # make plots
    _LOG.info("make and store distinct categories per sequence plots")
    plot_store_distinct_categories_per_sequence(
        trn_cats_per_seq_kdes=trn_cats_per_seq_kdes,
        syn_cats_per_seq_kdes=syn_cats_per_seq_kdes,
        trn_binned_cats_per_seq_cnts=trn_binned_cats_per_seq_cnts,
        syn_binned_cats_per_seq_cnts=syn_binned_cats_per_seq_cnts,
        acc_cats_per_seq=acc_cats_per_seq,
        workspace=workspace,
    )
    return acc_cats_per_seq


def _report_coherence_sequences_per_distinct_category(
    *,
    trn_coh: pd.DataFrame,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    # calculate sequences per distinct category
    _LOG.info("calculate sequences per distinct category for training")
    trn_seqs_per_cat_cnts, trn_seqs_per_top_cat_cnts, trn_top_cats, trn_n_seqs = (
        calculate_sequences_per_distinct_category(df=trn_coh, context_key=tgt_context_key)
    )
    _LOG.info("store sequences per distinct category artifacts for training")
    statistics.store_sequences_per_distinct_category_artifacts(
        seqs_per_cat_cnts=trn_seqs_per_cat_cnts,
        seqs_per_top_cat_cnts=trn_seqs_per_top_cat_cnts,
        top_cats=trn_top_cats,
        n_seqs=trn_n_seqs,
    )
    _LOG.info("calculate sequences per distinct category for synthetic")
    syn_seqs_per_cat_cnts, syn_seqs_per_top_cat_cnts, _, syn_n_seqs = calculate_sequences_per_distinct_category(
        df=syn_coh,
        context_key=tgt_context_key,
        top_cats=trn_top_cats,
    )

    # calculate per-column accuracy
    _LOG.info("calculate sequences per distinct category accuracy")
    acc_seqs_per_cat = calculate_sequences_per_distinct_category_accuracy(
        trn_seqs_per_top_cat_cnts=trn_seqs_per_top_cat_cnts,
        syn_seqs_per_top_cat_cnts=syn_seqs_per_top_cat_cnts,
    )
    _LOG.info("store sequences per distinct category accuracy")
    statistics.store_sequences_per_distinct_category_accuracy(accuracy=acc_seqs_per_cat)

    # make plots
    _LOG.info("make and store sequences per distinct category plots")
    plot_store_sequences_per_distinct_category(
        trn_seqs_per_cat_cnts=trn_seqs_per_cat_cnts,
        syn_seqs_per_cat_cnts=syn_seqs_per_cat_cnts,
        trn_seqs_per_top_cat_cnts=trn_seqs_per_top_cat_cnts,
        syn_seqs_per_top_cat_cnts=syn_seqs_per_top_cat_cnts,
        trn_n_seqs=trn_n_seqs,
        syn_n_seqs=syn_n_seqs,
        acc_seqs_per_cat=acc_seqs_per_cat,
        workspace=workspace,
    )

    return acc_seqs_per_cat


def _report_similarity(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
    workspace: TemporaryWorkspace,
    statistics: Statistics,
) -> tuple[np.float64 | None, np.float64, np.float64 | None, np.float64]:
    _LOG.info("calculate centroid similarities")
    sim_cosine_trn_hol, sim_cosine_trn_syn = _similarity.calculate_cosine_similarities(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )

    _LOG.info("calculate discriminator AUC")
    sim_auc_trn_hol, sim_auc_trn_syn = _similarity.calculate_discriminator_auc(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )

    _LOG.info("plot and store PCA similarity contours")
    pca_model, _, trn_pca, hol_pca = _similarity.plot_store_similarity_contours(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds, workspace=workspace
    )

    _LOG.info("store PCA model")
    statistics.store_pca_model(pca_model)

    _LOG.info("store training and holdout PCA-projected embeddings")
    statistics.store_trn_hol_pcas(trn_pca, hol_pca)

    return (
        sim_cosine_trn_hol,
        sim_cosine_trn_syn,
        sim_auc_trn_hol,
        sim_auc_trn_syn,
    )


def _report_distances(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
    workspace: TemporaryWorkspace,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    dcr_syn_trn, dcr_syn_hol, dcr_trn_hol = _distances.calculate_distances(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    _distances.plot_store_distances(dcr_syn_trn, dcr_syn_hol, dcr_trn_hol, workspace)
    return dcr_syn_trn, dcr_syn_hol, dcr_trn_hol
