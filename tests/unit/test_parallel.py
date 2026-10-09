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

import pandas as pd
import pytest
from joblib import Parallel, parallel_config
from joblib._parallel_backends import ThreadingBackend
from joblib.parallel import get_active_backend

from mostlyai.qa import _accuracy, _coherence, _parallel


def test_parallel_context_preserves_caller_settings():
    with parallel_config(backend="threading", n_jobs=2):
        with _parallel.parallel_context():
            backend, n_jobs = get_active_backend()
            assert isinstance(backend, ThreadingBackend)
            assert n_jobs == 2
        assert get_active_backend()[1] == 2


@pytest.mark.parametrize("cores, expected", [(1, 1), (4, 3), (64, 16)])
def test_default_worker_limit(monkeypatch, cores, expected):
    monkeypatch.setattr(_parallel, "cpu_count", lambda: cores)
    with _parallel.parallel_context():
        assert get_active_backend()[1] == expected


def test_accuracy_and_coherence_respect_outer_configuration(monkeypatch):
    observed = []

    def configured_parallel():
        parallel = Parallel()
        observed.append((type(parallel._backend), parallel.n_jobs))
        return parallel

    monkeypatch.setattr(_accuracy, "Parallel", configured_parallel)
    monkeypatch.setattr(_coherence, "Parallel", configured_parallel)
    data = pd.DataFrame({"tgt::a": pd.Categorical(["a", "b", "a"])})
    with parallel_config(backend="threading", n_jobs=2):
        counts, _ = _accuracy.calculate_bin_counts(data)
        assert counts["tgt::a"].sum() == 3
        result = _coherence.calculate_distinct_categories_per_sequence_accuracy(data, data)
        assert result["accuracy"].tolist() == [1.0]
    assert len(observed) == 3
    assert all(issubclass(backend, ThreadingBackend) and n_jobs == 2 for backend, n_jobs in observed)
