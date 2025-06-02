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
from pandas.core.dtypes.common import is_datetime64_dtype, is_numeric_dtype

from mostlyai.qa import _distances, _html_report, _similarity
from mostlyai.qa._accuracy import (
    bin_data,
    binning_data,
    calculate_bin_counts,
    calculate_bivariates,
    calculate_categorical_uni_counts,
    calculate_correlations,
    calculate_numeric_uni_kdes,
    calculate_trivariates,
    calculate_univariates,
    filter_biv_acc_for_plotting,
    filter_uni_acc_for_plotting,
    plot_store_accuracy_matrix,
    plot_store_bivariates,
    plot_store_correlation_matrices,
    plot_store_univariates,
)
from mostlyai.qa._coherence import (
    calculate_distinct_categories_per_sequence,
    calculate_distinct_categories_per_sequence_accuracy,
    calculate_sequences_per_distinct_category,
    calculate_sequences_per_distinct_category_accuracy,
    plot_store_distinct_categories_per_sequence,
    plot_store_sequences_per_distinct_category,
)
from mostlyai.qa._common import (
    CTX_COLUMN_PREFIX,
    NXT_COLUMN,
    REPORT_CREDITS,
    TGT_COLUMN_PREFIX,
    PrerequisiteNotMetError,
    ProgressCallback,
    ProgressCallbackWrapper,
    check_min_sample_size,
    determine_data_size,
)
from mostlyai.qa._filesystem import Statistics, TemporaryWorkspace
from mostlyai.qa._sampling import (
    prepare_data_for_accuracy,
    prepare_data_for_coherence,
    prepare_data_for_embeddings,
)
from mostlyai.qa.metrics import Accuracy, Distances, ModelMetrics, Similarity

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
    max_sample_size_coherence: int | None = None,
    max_sample_size_embeddings: int | None = None,
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
        max_sample_size_embeddings: The maximum sample size for embedding calculations.
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

        # warn if dtypes are inconsistent across datasets
        _warn_if_dtypes_inconsistent(syn_tgt_data, trn_tgt_data, hol_tgt_data)
        _warn_if_dtypes_inconsistent(syn_ctx_data, trn_ctx_data, hol_ctx_data)

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
            if trn_tgt_data.shape[1] == 0 or syn_tgt_data.shape[1] == 0:
                raise PrerequisiteNotMetError("Provided data has no columns.")
        except PrerequisiteNotMetError as err:
            _LOG.info(err)
            statistics.mark_early_exit()
            _html_report.store_early_exit_report(report_path)
            return report_path, None

        ## 0. PREPARE DATA ##

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

        trn = prepare_data_for_accuracy(
            df_tgt=trn_tgt_data,
            df_ctx=trn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            setup=setup,
        )
        _LOG.info(f"prepared training data for accuracy: {trn.shape}")
        if hol_tgt_data is not None:
            hol = prepare_data_for_accuracy(
                df_tgt=hol_tgt_data,
                df_ctx=hol_ctx_data,
                ctx_primary_key=ctx_primary_key,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_accuracy,
                setup=setup,
                ori_dtypes=trn.dtypes.to_dict(),
            )
            _LOG.info(f"prepared holdout data for accuracy: {hol.shape}")
            ori = pd.concat([trn, hol], axis=0, ignore_index=True)
        else:
            hol = None
            ori = trn
        progress.update(completed=5, total=100)

        syn = prepare_data_for_accuracy(
            df_tgt=syn_tgt_data,
            df_ctx=syn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            setup=setup,
            ori_dtypes=trn.dtypes.to_dict(),
        )
        _LOG.info(f"prepared synthetic data for accuracy: {syn.shape}")
        progress.update(completed=10, total=100)

        # do coherence analysis only if there are non-fk columns in the target data
        do_coherence = setup == "1:N" and len(trn_tgt_data.columns) > 1
        if do_coherence:
            ori_coh, ori_coh_bins = prepare_data_for_coherence(
                df_tgt=pd.concat([trn_tgt_data, hol_tgt_data]) if hol_tgt_data is not None else trn_tgt_data,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_coherence,
            )
            _LOG.info(f"prepared original data for coherence: {ori_coh.shape}")
            syn_coh, _ = prepare_data_for_coherence(
                df_tgt=syn_tgt_data,
                tgt_context_key=tgt_context_key,
                bins=ori_coh_bins,
                max_sample_size=max_sample_size_coherence,
            )
            _LOG.info(f"prepared synthetic data for coherence: {syn_coh.shape}")
            statistics.store_coherence_bins(bins=ori_coh_bins)
            _LOG.info("stored bins used for training data for coherence")
        progress.update(completed=15, total=100)

        syn_embeds, trn_embeds, hol_embeds = prepare_data_for_embeddings(
            syn_tgt_data=syn_tgt_data,
            trn_tgt_data=trn_tgt_data,
            hol_tgt_data=hol_tgt_data,
            syn_ctx_data=syn_ctx_data,
            trn_ctx_data=trn_ctx_data,
            hol_ctx_data=hol_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_embeddings,
        )
        _LOG.info(
            f"calculated embeddings: syn={syn_embeds.shape}, trn={trn_embeds.shape}, hol={hol_embeds.shape if hol_embeds is not None else None}"
        )
        progress.update(completed=20, total=100)

        ## 1. ACCURACY ##

        _LOG.info("report accuracy and correlations")
        acc_uni, acc_biv, acc_triv, corr_trn = _report_accuracy_and_correlations(
            ori=ori,
            syn=syn,
            statistics=statistics,
            workspace=workspace,
        )
        progress.update(completed=30, total=100)

        if do_coherence:
            _LOG.info("report sequences per distinct category")
            acc_seqs_per_cat = _report_coherence_sequences_per_distinct_category(
                ori_coh=ori_coh,
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
            _LOG.info("report distinct categories per sequence")
            acc_cats_per_seq = _report_coherence_distinct_categories_per_sequence(
                ori_coh=ori_coh,
                syn_coh=syn_coh,
                tgt_context_key=tgt_context_key,
                statistics=statistics,
                workspace=workspace,
            )
        else:
            acc_cats_per_seq = acc_seqs_per_cat = pd.DataFrame({"column": [], "accuracy": [], "accuracy_max": []})
        progress.update(completed=40, total=100)

        ## 2. SIMILARITY ##

        _LOG.info("report similarity")
        sim_cosine_trn_hol, sim_cosine_trn_syn, sim_auc_trn_hol, sim_auc_trn_syn = _report_similarity(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds if hol_embeds is not None else None,
            workspace=workspace,
        )
        progress.update(completed=60, total=100)

        ## 3. DISTANCES ##

        _LOG.info("calculate and plot distances")
        distances = _report_distances(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds if hol_embeds is not None else None,
            workspace=workspace,
        )
        progress.update(completed=90, total=100)

        ## 4. METRICS & REPORT ##

        _LOG.info("calculate metrics")
        metrics = _calculate_metrics(
            acc_uni=acc_uni,
            acc_biv=acc_biv,
            acc_triv=acc_triv,
            dcr_syn_trn=distances["dcr_syn_trn"],
            dcr_syn_hol=distances["dcr_syn_hol"],
            dcr_trn_hol=distances["dcr_trn_hol"],
            nndr_syn_trn=distances["nndr_syn_trn"],
            nndr_syn_hol=distances["nndr_syn_hol"],
            nndr_trn_hol=distances["nndr_trn_hol"],
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
            "tgt_columns": len([c for c in ori.columns if c.startswith(TGT_COLUMN_PREFIX)]),
            "ctx_columns": len([c for c in ori.columns if c.startswith(CTX_COLUMN_PREFIX)]),
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
            acc_biv=acc_biv,
            acc_triv=acc_triv,
            acc_cats_per_seq=acc_cats_per_seq,
            acc_seqs_per_cat=acc_seqs_per_cat,
            corr_trn=corr_trn,
        )
        progress.update(completed=100, total=100)
        _LOG.info(f"report stored at {report_path}")
        return report_path, metrics


def _warn_if_dtypes_inconsistent(syn_df: pd.DataFrame | None, trn_df: pd.DataFrame | None, hol_df: pd.DataFrame | None):
    dfs = [df for df in (syn_df, trn_df, hol_df) if df is not None]
    if not dfs:
        return
    common_columns = set.intersection(*[set(df.columns) for df in dfs])
    column_dtypes = {col: [df[col].dtype for df in dfs] for col in common_columns}
    inconsistent_columns = []
    for col, dtypes in column_dtypes.items():
        any_datetimes = any(is_datetime64_dtype(dtype) for dtype in dtypes)
        any_numbers = any(is_numeric_dtype(dtype) for dtype in dtypes)
        any_others = any(not is_datetime64_dtype(dtype) and not is_numeric_dtype(dtype) for dtype in dtypes)
        if sum([any_datetimes, any_numbers, any_others]) > 1:
            inconsistent_columns.append(col)
    if inconsistent_columns:
        warnings.warn(
            UserWarning(
                f"The column(s) {inconsistent_columns} have inconsistent data types across `syn`, `trn`, and `hol`. "
                "To achieve the most accurate results, please harmonize the data types of these inputs. "
                "Proceeding with a best-effort attempt..."
            )
        )


def _calculate_metrics(
    *,
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    acc_triv: pd.DataFrame,
    dcr_syn_trn: np.ndarray,
    dcr_syn_hol: np.ndarray | None,
    dcr_trn_hol: np.ndarray | None,
    nndr_syn_trn: np.ndarray,
    nndr_syn_hol: np.ndarray | None,
    nndr_trn_hol: np.ndarray | None,
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
    # trivariates
    if not acc_triv.empty:
        acc_trivariate = acc_triv.accuracy.mean()
        acc_trivariate_max = acc_triv.accuracy_max.mean()
    else:
        acc_trivariate = acc_trivariate_max = None
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
    acc_overall = np.mean([m for m in (acc_univariate, acc_bivariate, acc_trivariate, acc_coherence) if m is not None])
    acc_overall_max = np.mean(
        [m for m in (acc_univariate_max, acc_bivariate_max, acc_trivariate_max, acc_coherence_max) if m is not None]
    )
    accuracy = Accuracy(
        overall=acc_overall,
        univariate=acc_univariate,
        bivariate=acc_bivariate,
        trivariate=acc_trivariate,
        coherence=acc_coherence,
        overall_max=acc_overall_max,
        univariate_max=acc_univariate_max,
        bivariate_max=acc_bivariate_max,
        trivariate_max=acc_trivariate_max,
        coherence_max=acc_coherence_max,
    )
    similarity = Similarity(
        cosine_similarity_training_synthetic=sim_cosine_trn_syn,
        cosine_similarity_training_holdout=sim_cosine_trn_hol if sim_cosine_trn_hol is not None else None,
        discriminator_auc_training_synthetic=sim_auc_trn_syn,
        discriminator_auc_training_holdout=sim_auc_trn_hol if sim_auc_trn_hol is not None else None,
    )
    distances = Distances(
        ims_training=_distances.calculate_ims(dcr_syn_trn),
        ims_holdout=_distances.calculate_ims(dcr_syn_hol) if dcr_syn_hol is not None else None,
        ims_trn_hol=_distances.calculate_ims(dcr_trn_hol) if dcr_trn_hol is not None else None,
        dcr_training=_distances.calculate_dcr(dcr_syn_trn),
        dcr_holdout=_distances.calculate_dcr(dcr_syn_hol) if dcr_syn_hol is not None else None,
        dcr_trn_hol=_distances.calculate_dcr(dcr_trn_hol) if dcr_trn_hol is not None else None,
        dcr_share=_distances.calculate_dcr_share(dcr_syn_trn=dcr_syn_trn, dcr_syn_hol=dcr_syn_hol)
        if dcr_syn_hol is not None
        else None,
        nndr_training=_distances.calculate_nndr(nndr_syn_trn),
        nndr_holdout=_distances.calculate_nndr(nndr_syn_hol) if nndr_syn_hol is not None else None,
        nndr_trn_hol=_distances.calculate_nndr(nndr_trn_hol) if nndr_trn_hol is not None else None,
    )
    return ModelMetrics(
        accuracy=accuracy,
        similarity=similarity,
        distances=distances,
    )


def _report_accuracy_and_correlations(
    *,
    ori: pd.DataFrame,
    syn: pd.DataFrame,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # bin data
    ori_bin, syn_bin = binning_data(
        ori=ori,
        syn=syn,
        statistics=statistics,
    )

    # calculate correlations for original data
    corr_ori = calculate_correlations(binned=ori_bin)

    # store correlations for original data
    statistics.store_correlations(corr=corr_ori)

    # calculate correlations for synthetic data
    corr_syn = calculate_correlations(binned=syn_bin, corr_cols=corr_ori.columns)

    # plot correlations matrices
    plot_store_correlation_matrices(corr_ori=corr_ori, corr_syn=corr_syn, workspace=workspace)

    # calculate univariate accuracies
    acc_uni = calculate_univariates(ori_bin, syn_bin)

    # calculate bivariate accuracies
    acc_biv = calculate_bivariates(ori_bin, syn_bin)

    # calculate trivariate accuracies
    acc_triv = calculate_trivariates(ori_bin, syn_bin)

    # plot and store accuracy matrix
    plot_store_accuracy_matrix(
        acc_uni=acc_uni,
        acc_biv=acc_biv,
        workspace=workspace,
    )

    # filter columns for plotting
    acc_uni_plt = filter_uni_acc_for_plotting(acc_uni)
    acc_biv_plt = filter_biv_acc_for_plotting(acc_biv, corr_ori)
    ori = ori[acc_uni_plt["column"]]
    syn = syn[acc_uni_plt["column"]]
    acc_cols_plt = list(set(acc_uni["column"]) | set(acc_biv["col1"]) | set(acc_biv["col2"]))
    ori_bin = ori_bin[acc_cols_plt]
    syn_bin = syn_bin[acc_cols_plt]

    # store univariate and bivariate accuracies
    statistics.store_univariate_accuracies(acc_uni)
    statistics.store_bivariate_accuracies(acc_biv)
    statistics.store_trivariate_accuracies(acc_triv)

    # calculate KDEs for original
    ori_num_kdes = calculate_numeric_uni_kdes(ori)

    # store KDEs for original
    statistics.store_numeric_uni_kdes(ori_num_kdes)

    # calculate KDEs for synthetic
    syn_num_kdes = calculate_numeric_uni_kdes(syn, ori_num_kdes)

    # calculate categorical counts for original
    ori_cat_uni_cnts = calculate_categorical_uni_counts(df=ori, hash_rare_values=True)

    # store categorical counts for original
    statistics.store_categorical_uni_counts(ori_cat_uni_cnts)

    # calculate categorical counts for synthetic
    syn_cat_uni_cnts = calculate_categorical_uni_counts(
        df=syn,
        ori_col_counts=ori_cat_uni_cnts,
        hash_rare_values=False,
    )

    # calculate bin counts for original
    ori_bin_cnts_uni, ori_bin_cnts_biv = calculate_bin_counts(ori_bin)

    # store bin counts for original
    statistics.store_bin_counts(ori_cnts_uni=ori_bin_cnts_uni, ori_cnts_biv=ori_bin_cnts_biv)

    # calculate bin counts for synthetic
    syn_bin_cnts_uni, syn_bin_cnts_biv = calculate_bin_counts(binned=syn_bin)

    # plot univariate distributions
    plot_store_univariates(
        ori_num_kdes=ori_num_kdes,
        syn_num_kdes=syn_num_kdes,
        ori_cat_cnts=ori_cat_uni_cnts,
        syn_cat_cnts=syn_cat_uni_cnts,
        ori_cnts_uni=ori_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        acc_uni=acc_uni_plt,
        workspace=workspace,
        show_accuracy=True,
    )

    # plot bivariate distributions
    plot_store_bivariates(
        ori_cnts_uni=ori_bin_cnts_uni,
        syn_cnts_uni=syn_bin_cnts_uni,
        ori_cnts_biv=ori_bin_cnts_biv,
        syn_cnts_biv=syn_bin_cnts_biv,
        acc_biv=acc_biv_plt,
        workspace=workspace,
        show_accuracy=True,
    )

    return acc_uni, acc_biv, acc_triv, corr_ori


def _report_coherence_distinct_categories_per_sequence(
    *,
    ori_coh: pd.DataFrame,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    # calculate distinct categories per sequence
    _LOG.info("calculate distinct categories per sequence for training")
    ori_cats_per_seq = calculate_distinct_categories_per_sequence(df=ori_coh, context_key=tgt_context_key)
    _LOG.info("calculate distinct categories per sequence for synthetic")
    syn_cats_per_seq = calculate_distinct_categories_per_sequence(df=syn_coh, context_key=tgt_context_key)

    # bin distinct categories per sequence
    _LOG.info("bin distinct categories per sequence for training")
    ori_binned_cats_per_seq, bins = bin_data(ori_cats_per_seq, bins=10)
    _LOG.info("store distinct categories per sequence bins for training")
    statistics.store_distinct_categories_per_sequence_bins(bins=bins)
    _LOG.info("bin distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq, _ = bin_data(syn_cats_per_seq, bins=bins)

    # prepare KDEs for distribution (left) plots
    _LOG.info("calculate KDEs of distinct categories per sequence for training")
    ori_cats_per_seq_kdes = calculate_numeric_uni_kdes(df=ori_cats_per_seq)
    _LOG.info("store KDEs of distinct categories per sequence for training")
    statistics.store_distinct_categories_per_sequence_kdes(ori_kdes=ori_cats_per_seq_kdes)
    _LOG.info("calculate KDEs of distinct categories per sequence for synthetic")
    syn_cats_per_seq_kdes = calculate_numeric_uni_kdes(df=syn_cats_per_seq, ori_kdes=ori_cats_per_seq_kdes)

    # prepare counts for binned (right) plots
    _LOG.info("calculate counts of binned distinct categories per sequence for training")
    ori_binned_cats_per_seq_cnts = calculate_categorical_uni_counts(df=ori_binned_cats_per_seq, hash_rare_values=False)
    _LOG.info("store counts of binned distinct categories per sequence for training")
    statistics.store_binned_distinct_categories_per_sequence_counts(counts=ori_binned_cats_per_seq_cnts)
    _LOG.info("calculate counts of binned distinct categories per sequence for synthetic")
    syn_binned_cats_per_seq_cnts = calculate_categorical_uni_counts(df=syn_binned_cats_per_seq, hash_rare_values=False)

    # calculate per-column accuracy
    _LOG.info("calculate distinct categories per sequence accuracy")
    acc_cats_per_seq = calculate_distinct_categories_per_sequence_accuracy(
        ori_binned_cats_per_seq=ori_binned_cats_per_seq, syn_binned_cats_per_seq=syn_binned_cats_per_seq
    )
    _LOG.info("store distinct categories per sequence accuracy")
    statistics.store_distinct_categories_per_sequence_accuracy(accuracy=acc_cats_per_seq)

    # make plots
    _LOG.info("make and store distinct categories per sequence plots")
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
    ori_coh: pd.DataFrame,
    syn_coh: pd.DataFrame,
    tgt_context_key: str,
    statistics: Statistics,
    workspace: TemporaryWorkspace,
) -> pd.DataFrame:
    # calculate sequences per distinct category
    _LOG.info("calculate sequences per distinct category for training")
    ori_seqs_per_cat_cnts, ori_seqs_per_top_cat_cnts, ori_top_cats, ori_n_seqs = (
        calculate_sequences_per_distinct_category(df=ori_coh, context_key=tgt_context_key)
    )
    _LOG.info("store sequences per distinct category artifacts for training")
    statistics.store_sequences_per_distinct_category_artifacts(
        seqs_per_cat_cnts=ori_seqs_per_cat_cnts,
        seqs_per_top_cat_cnts=ori_seqs_per_top_cat_cnts,
        top_cats=ori_top_cats,
        n_seqs=ori_n_seqs,
    )
    _LOG.info("calculate sequences per distinct category for synthetic")
    syn_seqs_per_cat_cnts, syn_seqs_per_top_cat_cnts, _, syn_n_seqs = calculate_sequences_per_distinct_category(
        df=syn_coh,
        context_key=tgt_context_key,
        top_cats=ori_top_cats,
    )

    # calculate per-column accuracy
    _LOG.info("calculate sequences per distinct category accuracy")
    acc_seqs_per_cat = calculate_sequences_per_distinct_category_accuracy(
        ori_seqs_per_top_cat_cnts=ori_seqs_per_top_cat_cnts,
        syn_seqs_per_top_cat_cnts=syn_seqs_per_top_cat_cnts,
    )
    _LOG.info("store sequences per distinct category accuracy")
    statistics.store_sequences_per_distinct_category_accuracy(accuracy=acc_seqs_per_cat)

    # make plots
    _LOG.info("make and store sequences per distinct category plots")
    plot_store_sequences_per_distinct_category(
        ori_seqs_per_cat_cnts=ori_seqs_per_cat_cnts,
        syn_seqs_per_cat_cnts=syn_seqs_per_cat_cnts,
        ori_seqs_per_top_cat_cnts=ori_seqs_per_top_cat_cnts,
        syn_seqs_per_top_cat_cnts=syn_seqs_per_top_cat_cnts,
        ori_n_seqs=ori_n_seqs,
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
    _similarity.plot_store_similarity_contours(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds, workspace=workspace
    )

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
) -> dict[str, np.ndarray]:
    _LOG.info("calculate distances")
    distances = _distances.calculate_distances(syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds)
    _LOG.info("plot and store distances")
    _distances.plot_store_distances(distances, workspace)
    return distances
