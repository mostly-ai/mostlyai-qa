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
from typing import Protocol

import pandas as pd
from tqdm.auto import tqdm

from mostlyai.qa.filesystem import Statistics

_LOG = logging.getLogger(__name__)


ACCURACY_MAX_COLUMNS = 300  # should be an even number and greater than 100

MAX_UNIVARIATE_PLOTS = 300
MAX_BIVARIATE_TGT_PLOTS = 300
MAX_BIVARIATE_CTX_PLOTS = 60
MAX_BIVARIATE_NXT_PLOTS = 60

NA_BIN = "(n/a)"
OTHER_BIN = "(other)"
EMPTY_BIN = "(empty)"
RARE_BIN = "_RARE_"
MIN_RARE_CAT_PROTECTION = 5
MAX_ENGINE_RARE_CATEGORY_THRESHOLD = 7

CTX_COLUMN = "ctx"
TGT_COLUMN = "tgt"
NXT_COLUMN = "nxt"
DELIMITER = "⁝"
CTX_COLUMN_PREFIX = f"{CTX_COLUMN}{DELIMITER}"
TGT_COLUMN_PREFIX = f"{TGT_COLUMN}{DELIMITER}"
NXT_COLUMN_PREFIX = f"{NXT_COLUMN}{DELIMITER}"
COUNT_COLUMN = "Sequence Length"

REPORT_CREDITS = "Generated with <a href='https://github.com/mostly-ai/mostlyai-qa'>mostlyai-qa</a>."

CHARTS_COLORS = {
    "background": "rgba(0,0,0,0)",
    "original": "#666666",
    "synthetic": "#24db96",
    "difference": "#ff3300",
    "gap": "#ffeded",
}

CHARTS_FONTS = {
    "title": dict(size=12, color="black", family="objectivity, verdana"),
    "base": dict(size=8, color="#a3a3a3", family="objectivity, verdana"),
    "hover": dict(
        font_size=12,
        font_color="black",
        font_family="objectivity, verdana",
        bgcolor="white",
    ),
}


class PrerequisiteNotMetError(Exception):
    pass


class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int) -> None: ...


def add_tqdm(on_progress: ProgressCallback | None = None, description: str = "Processing") -> ProgressCallback:
    pbar = tqdm(desc=description, total=100)

    def _on_progress(current: int, total: int):
        if on_progress is not None:
            on_progress(current, total)
        pbar.update(current - pbar.n)

    return _on_progress


def check_min_sample_size(size: int, min: int, type: str) -> None:
    if size < min:
        raise PrerequisiteNotMetError(f"At least {min} rows are required, but only {size} were found for {type}.")


def check_statistics_prerequisite(statistics: Statistics) -> None:
    if statistics.is_early_exit():
        message = "No statistics available. Report will be skipped."
        _LOG.info(message)
        raise PrerequisiteNotMetError(message)


def determine_data_size(
    tgt_data: pd.DataFrame | None = None,
    ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
) -> int:
    if ctx_data is not None and ctx_primary_key is not None:
        return len(ctx_data[ctx_primary_key].unique())
    elif ctx_data is not None and not ctx_data.empty:
        return len(ctx_data)
    elif tgt_data is not None and tgt_context_key is not None:
        return len(tgt_data[tgt_context_key].unique())
    elif tgt_data is not None and not tgt_data.empty:
        return len(tgt_data)
    else:
        return 0
