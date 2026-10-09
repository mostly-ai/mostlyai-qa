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

from joblib import cpu_count, parallel_config
from joblib.parallel import get_active_backend


def parallel_context():
    """Preserve caller joblib settings, with up to 16 workers when unspecified."""
    _, n_jobs = get_active_backend()
    if n_jobs is None:
        n_jobs = min(16, max(1, cpu_count() - 1))
    return parallel_config(n_jobs=n_jobs)
