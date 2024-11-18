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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_dtype

from mostlyai.qa import distances, similarity, html_report
from mostlyai.qa.accuracy import (
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
from mostlyai.qa.sampling import calculate_embeddings, pull_data_for_accuracy, pull_data_for_embeddings
from mostlyai.qa.common import (
    determine_data_size,
    ProgressCallback,
    PrerequisiteNotMetError,
    check_min_sample_size,
    add_tqdm,
    NXT_COLUMN,
    CTX_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
    REPORT_CREDITS,
)
from mostlyai.qa.filesystem import Statistics, TemporaryWorkspace

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
    report_extra_info: str = "",
    max_sample_size_accuracy: int | None = None,
    max_sample_size_embeddings: int | None = None,
    statistics_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, dict | None]:
    """
    Generate HTML report and metrics for comparing synthetic and original data samples.

    Args:
        syn_tgt_data: Synthetic samples
        trn_tgt_data: Training samples
        hol_tgt_data: Holdout samples
        syn_ctx_data: Synthetic context samples
        trn_ctx_data: Training context samples
        hol_ctx_data: Holdout context samples
        ctx_primary_key: Column within the context data that contains the primary key
        tgt_context_key: Column within the target data that contains the key to link to the context
        report_path: Path of where to store the HTML report
        report_title: Title of the HTML report
        report_subtitle: Subtitle of the HTML report
        report_credits: Credits of the HTML report
        report_extra_info: Extra details to be included to the HTML report
        max_sample_size_accuracy: Max sample size for accuracy
        max_sample_size_embeddings: Max sample size for embeddings (similarity & distances)
        statistics_path: Path of where to store the statistics to be used by `report_from_statistics`
        on_progress: A custom progress callback
    Returns:
        1. Path to the HTML report
        2. Dictionary of calculated metrics:
        - `accuracy`:  # Accuracy is defined as (100% - Total Variation Distance), for each distribution, and then averaged across.
          - `overall`: Overall accuracy of synthetic data, i.e. average across univariate, bivariate and coherence.
          - `univariate`: Average accuracy of discretized univariate distributions.
          - `bivariate`: Average accuracy of discretized bivariate distributions.
          - `coherence`: Average accuracy of discretized coherence distributions. Only applicable for sequential data.
          - `overall_max`: Expected overall accuracy of a same-sized holdout. Serves as reference for `overall`.
          - `univariate_max`: Expected univariate accuracy of a same-sized holdout. Serves as reference for `univariate`.
          - `bivariate_max`: Expected bivariate accuracy of a same-sized holdout. Serves as reference for `bivariate`.
          - `coherence_max`: Expected coherence accuracy of a same-sized holdout. Serves as reference for `coherence`.
        - `similarity`:  # All similarity metrics are calculated within an embedding space.
            - `cosine_similarity_training_synthetic`: Cosine similarity between training and synthetic centroids.
            - `cosine_similarity_training_holdout`: Cosine similarity between training and holdout centroids. Serves as reference for `cosine_similarity_training_synthetic`.
            - `discriminator_auc_training_synthetic`: Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples.
            - `discriminator_auc_training_holdout`: Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as reference for `discriminator_auc_training_synthetic`.
        - `distances`:  # All distance metrics are calculated within an embedding space. An equal number of training and holdout samples is considered.
            - `ims_training`: Share of synthetic samples that are identical to a training sample.
            - `ims_holdout`: Share of synthetic samples that are identical to a holdout sample. Serves as reference for `ims_training`.
            - `dcr_training`: Average L2 nearest-neighbor distance between synthetic and training samples.
            - `dcr_holdout`: Average L2 nearest-neighbor distance between synthetic and holdout samples. Serves as reference for `dcr_training`.
            - `dcr_share`: Share of synthetic samples that are closer to a training sample than to a holdout sample. This shall not be significantly larger than 50\%.
    """

    with TemporaryWorkspace() as workspace:
        on_progress = add_tqdm(on_progress, description="Creating report")
        on_progress(current=0, total=100)

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
                check_min_sample_size(trn_sample_size, 10, "holdout")
        except PrerequisiteNotMetError as err:
            _LOG.info(err)
            statistics.mark_early_exit()
            html_report.store_early_exit_report(report_path)
            on_progress(current=100, total=100)
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
        on_progress(current=10, total=100)

        _LOG.info("prepare training data for accuracy started")
        trn = pull_data_for_accuracy(
            df_tgt=trn_tgt_data,
            df_ctx=trn_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            max_sample_size=max_sample_size_accuracy,
            setup=setup,
        )
        on_progress(current=20, total=100)

        # coerce dtypes to match the original training data dtypes
        for col in trn:
            if is_numeric_dtype(trn[col]):
                syn[col] = pd.to_numeric(syn[col], errors="coerce")
            elif is_datetime64_dtype(trn[col]):
                syn[col] = pd.to_datetime(syn[col], errors="coerce")
            syn[col] = syn[col].astype(trn[col].dtype)

        _LOG.info("report accuracy and correlations")
        acc_uni, acc_biv, corr_trn = report_accuracy_and_correlations(
            trn=trn,
            syn=syn,
            statistics=statistics,
            workspace=workspace,
        )
        on_progress(current=30, total=100)

        # ensure that embeddings are all of equal size for a fair 3-way comparison
        max_sample_size_embeddings = min(syn_sample_size, trn_sample_size)
        if hol_sample_size != 0:
            max_sample_size_embeddings = min(max_sample_size_embeddings, hol_sample_size)

        # calculate embeddings
        syn_embeds = calculate_embeddings(
            pull_data_for_embeddings(
                df_tgt=syn_tgt_data,
                df_ctx=syn_ctx_data,
                ctx_primary_key=ctx_primary_key,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_embeddings,
            )
        )
        _LOG.info(f"calculated embeddings for synthetic {syn_embeds.shape}")
        on_progress(current=50, total=100)
        trn_embeds = calculate_embeddings(
            pull_data_for_embeddings(
                df_tgt=trn_tgt_data,
                df_ctx=trn_ctx_data,
                ctx_primary_key=ctx_primary_key,
                tgt_context_key=tgt_context_key,
                max_sample_size=max_sample_size_embeddings,
            )
        )
        _LOG.info(f"calculated embeddings for training {trn_embeds.shape}")
        on_progress(current=60, total=100)
        if hol_tgt_data is not None:
            hol_embeds = calculate_embeddings(
                pull_data_for_embeddings(
                    df_tgt=hol_tgt_data,
                    df_ctx=hol_ctx_data,
                    ctx_primary_key=ctx_primary_key,
                    tgt_context_key=tgt_context_key,
                    max_sample_size=max_sample_size_embeddings,
                )
            )
            _LOG.info(f"calculated embeddings for holdout {hol_embeds.shape}")
        else:
            hol_embeds = None
        on_progress(current=70, total=100)

        _LOG.info("report similarity")
        sim_cosine_trn_hol, sim_cosine_trn_syn, sim_auc_trn_hol, sim_auc_trn_syn = report_similarity(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds,
            workspace=workspace,
            statistics=statistics,
        )
        on_progress(current=80, total=100)

        _LOG.info("report distances")
        dcr_trn, dcr_hol = report_distances(
            syn_embeds=syn_embeds,
            trn_embeds=trn_embeds,
            hol_embeds=hol_embeds,
            workspace=workspace,
        )
        on_progress(current=90, total=100)

        metrics = calculate_metrics(
            acc_uni=acc_uni,
            acc_biv=acc_biv,
            dcr_trn=dcr_trn,
            dcr_hol=dcr_hol,
            sim_cosine_trn_hol=sim_cosine_trn_hol,
            sim_cosine_trn_syn=sim_cosine_trn_syn,
            sim_auc_trn_hol=sim_auc_trn_hol,
            sim_auc_trn_syn=sim_auc_trn_syn,
        )
        on_progress(current=95, total=100)

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
            "report_extra_info": report_extra_info,
        }
        statistics.store_meta(meta=meta)

        html_report.store_report(
            report_path=report_path,
            report_type="model_report",
            workspace=workspace,
            metrics=metrics,
            meta=meta,
            acc_uni=acc_uni,
            acc_biv=acc_biv,
            corr_trn=corr_trn,
        )
        on_progress(current=100, total=100)
        return report_path, metrics


def calculate_metrics(
    *,
    acc_uni: pd.DataFrame | None = None,
    acc_biv: pd.DataFrame | None = None,
    dcr_trn: np.ndarray | None = None,
    dcr_hol: np.ndarray | None = None,
    sim_cosine_trn_hol: np.float64 | None = None,
    sim_cosine_trn_syn: np.float64 | None = None,
    sim_auc_trn_hol: np.float64 | None = None,
    sim_auc_trn_syn: np.float64 | None = None,
) -> dict:
    # round all metrics
    precision_accuracy = 3
    precision_distances = 3
    precision_similarity_cosine = 7
    precision_similarity_auc = 3

    do_accuracy = acc_uni is not None and acc_biv is not None
    do_distances = dcr_trn is not None
    do_similarity = sim_cosine_trn_syn is not None

    metrics = {}
    if do_accuracy:
        # univariates
        acc_univariate = acc_uni.accuracy.mean().round(precision_accuracy)
        acc_univariate_max = acc_uni.accuracy_max.mean().round(precision_accuracy)
        # bivariates
        acc_tgt_ctx = acc_biv.loc[acc_biv.type != NXT_COLUMN]
        if not acc_tgt_ctx.empty:
            acc_bivariate = acc_tgt_ctx.accuracy.mean().round(precision_accuracy)
            acc_bivariate_max = acc_tgt_ctx.accuracy_max.mean().round(precision_accuracy)
        else:
            acc_bivariate = acc_bivariate_max = None
        # coherence
        acc_nxt = acc_biv.loc[acc_biv.type == NXT_COLUMN]
        if not acc_nxt.empty:
            acc_coherence = acc_nxt.accuracy.mean().round(precision_accuracy)
            acc_coherence_max = acc_nxt.accuracy_max.mean().round(precision_accuracy)
        else:
            acc_coherence = acc_coherence_max = None
        # calculate overall
        acc_overall_elems = (acc_univariate, acc_bivariate, acc_coherence)
        acc_overall_max_elems = (acc_univariate_max, acc_bivariate_max, acc_coherence_max)
        acc_overall = np.mean([m for m in acc_overall_elems if m is not None]).round(precision_accuracy)
        acc_overall_max = np.mean([m for m in acc_overall_max_elems if m is not None]).round(precision_accuracy)
        metrics["accuracy"] = {
            "overall": acc_overall,
            "univariate": acc_univariate,
            "bivariate": acc_bivariate,
            "coherence": acc_coherence,
            "overall_max": acc_overall_max,
            "univariate_max": acc_univariate_max,
            "bivariate_max": acc_bivariate_max,
            "coherence_max": acc_coherence_max,
        }
    if do_similarity:
        metrics["similarity"] = {
            "cosine_similarity_training_synthetic": round(sim_cosine_trn_syn, precision_similarity_cosine),
            "discriminator_auc_training_synthetic": round(sim_auc_trn_syn, precision_similarity_auc),
        }
        if sim_cosine_trn_hol is not None:
            metrics["similarity"]["cosine_similarity_training_holdout"] = round(
                sim_cosine_trn_hol, precision_similarity_cosine
            )
        else:
            metrics["similarity"]["cosine_similarity_training_holdout"] = None
        if sim_auc_trn_hol is not None:
            metrics["similarity"]["discriminator_auc_training_holdout"] = round(
                sim_auc_trn_hol, precision_similarity_auc
            )
        else:
            metrics["similarity"]["discriminator_auc_training_holdout"] = None
    if do_distances:
        metrics["distances"] = {
            "ims_training": round((dcr_trn <= 1e-6).mean(), precision_distances),
            "dcr_training": round(dcr_trn.mean(), precision_distances),
        }
        if dcr_hol is not None:
            metrics["distances"]["ims_holdout"] = round((dcr_hol <= 1e-6).mean(), precision_distances)
            metrics["distances"]["dcr_holdout"] = round(dcr_hol.mean(), precision_distances)
            metrics["distances"]["dcr_share"] = np.mean(dcr_trn < dcr_hol) + np.mean(dcr_trn == dcr_hol) / 2
        else:
            metrics["distances"]["ims_holdout"] = None
            metrics["distances"]["dcr_holdout"] = None
            metrics["distances"]["dcr_share"] = None
    return metrics


def report_accuracy_and_correlations(
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


def report_similarity(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
    workspace: TemporaryWorkspace,
    statistics: Statistics,
) -> tuple[np.float64 | None, np.float64, np.float64 | None, np.float64]:
    _LOG.info("calculate centroid similarities")
    sim_cosine_trn_hol, sim_cosine_trn_syn = similarity.calculate_cosine_similarities(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )

    _LOG.info("calculate discriminator AUC")
    sim_auc_trn_hol, sim_auc_trn_syn = similarity.calculate_discriminator_auc(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )

    _LOG.info("plot and store PCA similarity contours")
    pca_model, _, trn_pca, hol_pca = similarity.plot_store_similarity_contours(
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


def report_distances(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
    workspace: TemporaryWorkspace,
) -> tuple[np.ndarray, np.ndarray | None]:
    dcr_trn, dcr_hol = distances.calculate_distances(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    distances.plot_store_distances(dcr_trn, dcr_hol, workspace)
    return dcr_trn, dcr_hol
