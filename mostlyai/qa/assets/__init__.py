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

import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_HTML_ASSET_FILES = [
    "bootstrap-5.3.3.bundle.min.js",
    "bootstrap-5.3.3.min.css",
    "plotly-3.0.1.min.js",
    "explainer.svg",
    "info.svg",
]
HTML_ASSETS_PATH = _MODULE_DIR / "html"
HTML_REPORT_TEMPLATE = "report_template.html"
HTML_REPORT_EARLY_EXIT = "report_early_exit.html"


def read_html_assets() -> dict[str, str]:
    return {fn: Path(HTML_ASSETS_PATH / fn).read_text(encoding='utf-8') for fn in _HTML_ASSET_FILES}


def load_tokenizer():
    from transformers import GPT2Tokenizer

    return GPT2Tokenizer.from_pretrained(_MODULE_DIR / "tokenizers" / "transformers" / "gpt2")


def load_embedder():
    """
    Load the embedder model.
    Can deal with read-only cache folder by attempting to download the model if it is not locally available.
    Users can set MOSTLY_HF_HOME environment variable to override the default cache folder.

    Note that this method can take significant time to load the model. Thus, it is recommended to call this method once and reuse the returned object.
    """
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_folder = os.getenv("MOSTLY_HF_HOME")
    try:
        # First try loading from local cache
        return SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_folder, local_files_only=True)
    except Exception:
        # If not found in cache, attempt downloading
        return SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_folder, local_files_only=False)


__all__ = ["load_embedder"]
