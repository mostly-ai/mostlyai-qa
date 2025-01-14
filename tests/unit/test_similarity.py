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

import numpy as np

from mostlyai.qa._similarity import calculate_cosine_similarities, calculate_discriminator_auc
from mostlyai.qa._sampling import calculate_embeddings


def test_calculate_embeddings():
    trn = ["apple recipe", "car engine repair", "apple recipe"]
    # semantically close synthetic data
    syn_close = ["apple pie", "car maintenance"]
    # semantically distant synthetic data
    syn_distant = ["quantum physics theory", "deep space exploration"]

    trn_embeds = calculate_embeddings(trn)
    syn_close_embeds = calculate_embeddings(syn_close)
    syn_distant_embeds = calculate_embeddings(syn_distant)
    assert np.all(trn_embeds[0] == trn_embeds[2])  # check that we retain row order

    # check that syn_close is closer to trn than syn_distant
    def centroid_l2(a, b):
        return np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0), ord=2)

    assert centroid_l2(trn_embeds, syn_close_embeds) < centroid_l2(trn_embeds, syn_distant_embeds)


def test_calculate_centroid_similarities():
    syn_embeds = np.array([[0, -1], [1, 0], [1, -1]])
    trn_embeds = np.array([[1, 0], [0, 1], [1, 1]])
    hol_embeds = np.array([[1, 0], [0, 1], [1, 1]])
    sim_trn_hol, sim_trn_syn = calculate_cosine_similarities(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    np.testing.assert_allclose(sim_trn_hol, 1.0, atol=1e-10)
    np.testing.assert_allclose(sim_trn_syn, 0.0, atol=1e-10)


def test_calculate_discriminator_auc():
    syn_embeds = np.random.rand(1000, 100)
    trn_embeds = np.random.rand(1000, 100)
    hol_embeds = np.random.rand(1000, 100)
    sim_trn_hol, sim_trn_syn = calculate_discriminator_auc(
        syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds
    )
    np.testing.assert_allclose(sim_trn_hol, 0.5, atol=0.1)
    np.testing.assert_allclose(sim_trn_syn, 0.5, atol=0.1)
