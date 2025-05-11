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

from mostlyai.qa import _accuracy, _html_report, _distances, _similarity
from mostlyai.qa._common import CTX_COLUMN_PREFIX, TGT_COLUMN_PREFIX
from mostlyai.qa.reporting import _calculate_metrics
import pandas as pd


def test_generate_store_report(tmp_path, cols, workspace):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "_.", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, hol.columns, syn.columns = columns, columns, columns
    trn["nxt::dt"], hol["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], hol["tgt::dt"], syn["tgt::dt"]
    acc_trn, bins = _accuracy.bin_data(trn, 3)
    acc_syn, _ = _accuracy.bin_data(syn, bins)
    acc_uni = _accuracy.calculate_univariates(acc_trn, acc_syn)
    acc_biv = _accuracy.calculate_bivariates(acc_trn, acc_syn)
    acc_triv = _accuracy.calculate_trivariates(acc_trn, acc_syn)
    acc_cats_per_seq = pd.DataFrame({"column": acc_uni["column"], "accuracy": 0.5, "accuracy_max": 0.5})
    acc_seqs_per_cat = pd.DataFrame({"column": acc_uni["column"], "accuracy": 0.5, "accuracy_max": 0.5})
    corr_trn = _accuracy.calculate_correlations(acc_trn)
    syn_embeds, trn_embeds, hol_embeds = _distances.encode_data(
        syn=syn,
        trn=trn,
        hol=hol,
    )
    sim_cosine_trn_hol, sim_cosine_trn_syn = _similarity.calculate_cosine_similarities(
        syn_embeds=syn_embeds.values,
        trn_embeds=trn_embeds.values,
        hol_embeds=hol_embeds.values,
    )
    sim_auc_trn_hol, sim_auc_trn_syn = _similarity.calculate_discriminator_auc(
        syn_embeds=syn_embeds.values,
        trn_embeds=trn_embeds.values,
        hol_embeds=hol_embeds.values,
    )
    distances = _distances.calculate_distances(syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds)

    # simulate created plots
    plot_paths = (
        list(workspace.get_figure_paths("univariate", acc_uni[["column"]]).values())
        + list(workspace.get_figure_paths("bivariate", acc_biv[["col1", "col2"]]).values())
        + [workspace.get_unique_figure_path("accuracy_matrix")]
        + [workspace.get_unique_figure_path("correlation_matrices")]
        + [workspace.get_unique_figure_path("similarity_pca")]
        + [workspace.get_unique_figure_path("distances_dcr")]
        + list(workspace.get_figure_paths("distinct_categories_per_sequence", acc_cats_per_seq[["column"]]).values())
        + list(workspace.get_figure_paths("sequences_per_distinct_category", acc_seqs_per_cat[["column"]]).values())
    )
    for path in plot_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<div></div>", encoding="utf-8")

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
        "rows_original": trn.shape[0],
        "rows_synthetic": syn.shape[0],
        "tgt_columns": len([c for c in trn.columns if c.startswith(TGT_COLUMN_PREFIX)]),
        "ctx_columns": len([c for c in trn.columns if c.startswith(CTX_COLUMN_PREFIX)]),
    }

    report_path = tmp_path / "report.html"
    _html_report.store_report(
        report_path=report_path,
        workspace=workspace,
        report_type="model_report",
        metrics=metrics,
        meta=meta,
        acc_uni=acc_uni,
        acc_biv=acc_biv,
        acc_triv=acc_triv,
        corr_trn=corr_trn,
        acc_cats_per_seq=acc_cats_per_seq,
        acc_seqs_per_cat=acc_seqs_per_cat,
    )
    assert report_path.exists()


def test_summarize_accuracies_by_column(tmp_path, cols):
    trn, hol, syn = cols
    # apply ctx::/tgt:: prefixes and create nxt:: columns
    prefixes = ["ctx::", "_.", "tgt::"]
    columns = [f"{p}{c}" for p, c in zip(prefixes, trn.columns)]
    trn.columns, syn.columns = columns, columns
    trn["nxt::dt"], syn["nxt::dt"] = trn["tgt::dt"], syn["tgt::dt"]
    trn, bins = _accuracy.bin_data(trn, 3)
    syn, _ = _accuracy.bin_data(syn, bins)
    acc_uni = _accuracy.calculate_univariates(trn, syn)
    acc_biv = _accuracy.calculate_bivariates(trn, syn)
    acc_triv = _accuracy.calculate_trivariates(trn, syn)
    acc_cats_per_seq = pd.DataFrame({"column": acc_uni["column"], "accuracy": 0.5, "accuracy_max": 0.5})
    acc_seqs_per_cat = pd.DataFrame({"column": acc_uni["column"], "accuracy": 0.5, "accuracy_max": 0.5})
    tbl_acc = _html_report.summarize_accuracies_by_column(
        acc_uni, acc_biv, acc_triv, acc_cats_per_seq, acc_seqs_per_cat
    )
    assert (tbl_acc["univariate"] >= 0.5).all()
    assert (tbl_acc["bivariate"] >= 0.5).all()
    assert (tbl_acc["coherence"] >= 0.5).all()
    assert tbl_acc.shape[0] == len([c for c in trn if c.startswith("tgt::")])
