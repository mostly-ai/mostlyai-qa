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

import pandas as pd
import pytest

from mostlyai.qa._common import TGT_COLUMN_PREFIX
from mostlyai.qa._distances import (
    calculate_distances,
    plot_store_distances,
)
from mostlyai.qa._sampling import calculate_embeddings
from mostlyai.qa.assets import load_embedder


@pytest.fixture()
def too_many_cols():
    categories = ["A", "B", "C", "D", "E", "F", "G"]
    col = categories * 50
    df = pd.DataFrame({f"{TGT_COLUMN_PREFIX}cat_{i}": col for i in range(1100)})
    return df, df, df


@pytest.fixture()
def cat_with_rare_and_none():
    common = [f"common_{i}" for i in range(20)] + [None]
    rare = [f"rare_{i}" for i in range(100)]
    col = common * 10 + rare
    ser = pd.Series(col)
    return ser, ser, ser


def test_calculate_distances():
    n = 10
    embedder = load_embedder()
    syn_embeds = calculate_embeddings(["a 0 1.0"] * n, embedder=embedder)
    trn_embeds = calculate_embeddings(["a 0 0.0"] * n, embedder=embedder)
    hol_embeds = calculate_embeddings(["a 0 1.0"] * n, embedder=embedder)
    dcr_syn_trn, dcr_syn_hol, dcr_trn_hol = calculate_distances(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    assert len(dcr_syn_trn) == n
    assert len(dcr_syn_hol) == n
    assert len(dcr_trn_hol) == n
    assert dcr_syn_trn.min() > 0
    assert dcr_syn_hol.max() == 0

    # test specifically that near matches do not report a distance of 0 due to rounding
    syn_embeds = calculate_embeddings(["a 0.0002"] * n, embedder=embedder)
    trn_embeds = calculate_embeddings(["a 0.0001"] * n, embedder=embedder)
    hol_embeds = calculate_embeddings(["a 0.0001"] * n, embedder=embedder)
    dcr_syn_trn, dcr_syn_hol, dcr_trn_hol = calculate_distances(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    assert dcr_syn_hol.min() > 0
    assert dcr_trn_hol.min() == 0


def test_plot_store_dcr(workspace):
    embedder = load_embedder()
    embeds = calculate_embeddings(["a 0.0002"] * 100, embedder=embedder)
    dcr_syn_trn, dcr_syn_hol, dcr_trn_hol = calculate_distances(syn_embeds=embeds, trn_embeds=embeds, hol_embeds=embeds)
    plot_store_distances(dcr_syn_trn, dcr_syn_hol, dcr_trn_hol, workspace)
    output_dir = workspace.workspace_dir / "figures"
    assert len(list(output_dir.glob("*.html"))) == 1
