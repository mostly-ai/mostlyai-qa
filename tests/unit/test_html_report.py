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

import mostlyai.qa.accuracy
from mostlyai.qa import accuracy, html_report, distances
from mostlyai.qa.common import CTX_COLUMN_PREFIX, TGT_COLUMN_PREFIX
from mostlyai.qa.report import calculate_metrics
from mostlyai.qa.similarity import (
    calculate_cosine_similarities,
    calculate_discriminator_auc,
)
from mostlyai.qa.sampling import calculate_embeddings, pull_data_for_embeddings


def test_generate_store_report(tmp_path, cols, workspace):
    trn, hol, syn = cols
    # apply ctx⁝/tgt⁝ prefixes and create nxt⁝ columns
    prefixes = ["ctx⁝", "_.", "tgt⁝"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, hol.columns, syn.columns = columns, columns, columns
    trn["nxt⁝dt"], hol["nxt⁝dt"], syn["nxt⁝dt"] = trn["tgt⁝dt"], hol["tgt⁝dt"], syn["tgt⁝dt"]
    acc_trn, bins = mostlyai.qa.accuracy.bin_data(trn, 3)
    acc_syn, _ = mostlyai.qa.accuracy.bin_data(syn, bins)
    acc_uni = accuracy.calculate_univariates(acc_trn, acc_syn)
    acc_biv = accuracy.calculate_bivariates(acc_trn, acc_syn)
    corr_trn = mostlyai.qa.accuracy.calculate_correlations(acc_trn)
    syn_embeds = calculate_embeddings(pull_data_for_embeddings(df_tgt=syn))
    trn_embeds = calculate_embeddings(pull_data_for_embeddings(df_tgt=trn))
    hol_embeds = calculate_embeddings(pull_data_for_embeddings(df_tgt=hol))
    sim_cosine_trn_hol, sim_cosine_trn_syn = calculate_cosine_similarities(
        syn_embeds=syn_embeds,
        trn_embeds=trn_embeds,
        hol_embeds=hol_embeds,
    )
    sim_auc_trn_hol, sim_auc_trn_syn = calculate_discriminator_auc(
        syn_embeds=syn_embeds,
        trn_embeds=trn_embeds,
        hol_embeds=hol_embeds,
    )
    dcr_trn, dcr_hol = distances.calculate_distances(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )

    # simulate created plots
    plot_paths = (
        list(workspace.get_figure_paths("univariate", acc_uni[["column"]]).values())
        + list(workspace.get_figure_paths("bivariate", acc_biv[["col1", "col2"]]).values())
        + [workspace.get_unique_figure_path("accuracy_matrix")]
        + [workspace.get_unique_figure_path("correlation_matrices")]
        + [workspace.get_unique_figure_path("similarity_pca")]
        + [workspace.get_unique_figure_path("distances_dcr")]
    )
    for path in plot_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<div></div>")

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

    meta = {
        "rows_original": trn.shape[0],
        "rows_synthetic": syn.shape[0],
        "tgt_columns": len([c for c in trn.columns if c.startswith(TGT_COLUMN_PREFIX)]),
        "ctx_columns": len([c for c in trn.columns if c.startswith(CTX_COLUMN_PREFIX)]),
    }

    report_path = tmp_path / "report.html"
    html_report.store_report(
        report_path=report_path,
        workspace=workspace,
        report_type="model_report",
        metrics=metrics,
        meta=meta,
        acc_uni=acc_uni,
        acc_biv=acc_biv,
        corr_trn=corr_trn,
    )
    assert report_path.exists()


def test_summarize_accuracies_by_column(tmp_path, cols):
    trn, hol, syn = cols
    # apply ctx⁝/tgt⁝ prefixes and create nxt⁝ columns
    prefixes = ["ctx⁝", "_.", "tgt⁝"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt⁝dt"], syn["nxt⁝dt"] = trn["tgt⁝dt"], syn["tgt⁝dt"]
    trn, bins = mostlyai.qa.accuracy.bin_data(trn, 3)
    syn, _ = mostlyai.qa.accuracy.bin_data(syn, bins)
    uni_acc = accuracy.calculate_univariates(trn, syn)
    biv_acc = accuracy.calculate_bivariates(trn, syn)
    tbl_acc = html_report.summarize_accuracies_by_column(uni_acc, biv_acc)
    assert (tbl_acc["univariate"] >= 0.5).all()
    assert (tbl_acc["bivariate"] >= 0.5).all()
    assert (tbl_acc["coherence"] >= 0.5).all()
    assert tbl_acc.shape[0] == len([c for c in trn if c.startswith("tgt⁝")])
