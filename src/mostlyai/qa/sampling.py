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

#  Copyright 2024 MOSTLY AI Solutions MP GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
import random
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from mostlyai.qa.common import (
    CTX_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
    NXT_COLUMN_PREFIX,
    COUNT_COLUMN,
    ACCURACY_MAX_COLUMNS,
)
from mostlyai.qa.assets import load_embedder, load_tokenizer


_LOG = logging.getLogger(__name__)


def pull_data_for_accuracy(
    *,
    df_tgt: pd.DataFrame,
    df_ctx: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    max_sample_size: int | None = None,
    setup: str | None = None,
) -> pd.DataFrame:
    """
    Prepare single dataset for accuracy report.
    """

    # keys must be provided if df_ctx provided
    assert df_ctx is None or (ctx_primary_key is not None and tgt_context_key is not None)
    assert tgt_context_key is None or tgt_context_key in df_tgt.columns
    assert ctx_primary_key is None or ctx_primary_key in df_ctx.columns
    assert setup is None or setup in ["1:1", "1:N"]

    key = "__KEY"

    if df_ctx is not None:
        # explicit context
        df_ctx = df_ctx.sample(frac=1).head(max_sample_size)
        df_ctx = df_ctx.rename(columns={ctx_primary_key: tgt_context_key}).reset_index(drop=True)
        df_tgt = df_tgt.merge(df_ctx[tgt_context_key], on=tgt_context_key, how="inner").reset_index(drop=True)
    elif tgt_context_key is not None:
        # implicit context
        df_ctx = df_tgt[[tgt_context_key]].drop_duplicates()
        df_ctx = df_ctx.sample(frac=1).head(max_sample_size).reset_index(drop=True)
        df_tgt = df_tgt.merge(df_ctx[tgt_context_key], on=tgt_context_key, how="inner").reset_index(drop=True)
    else:
        # no context; flat table
        tgt_context_key = key
        df_tgt = df_tgt.sample(frac=1).head(max_sample_size).reset_index(drop=True)
        df_tgt[key] = range(len(df_tgt))
        df_ctx = df_tgt[[key]]

    # consistently use "__KEY" as key column
    df_ctx = df_ctx.rename(columns={tgt_context_key: key})
    df_tgt = df_tgt.rename(columns={tgt_context_key: key})

    # limit to ACCURACY_MAX_COLUMNS columns
    df_tgt = df_tgt[[key] + [c for c in sorted(df_tgt.columns) if c != key][:ACCURACY_MAX_COLUMNS]]
    df_ctx = df_ctx[[key] + [c for c in sorted(df_ctx.columns) if c != key][:ACCURACY_MAX_COLUMNS]]

    # count records
    df_cnt = df_tgt.groupby(key).size().to_frame(COUNT_COLUMN).reset_index()
    df_cnt.columns = [TGT_COLUMN_PREFIX + c if c != key else c for c in df_cnt.columns]

    # pick two random consecutive rows (if sequential)
    df_tgt, df_nxt = sample_two_consecutive_rows(df_tgt, key)

    # prefix column names to avoid column name conflicts when merging
    df_ctx.columns = [CTX_COLUMN_PREFIX + c if c != key else c for c in df_ctx.columns]
    df_tgt.columns = [TGT_COLUMN_PREFIX + c if c != key else c for c in df_tgt.columns]
    df_nxt.columns = [NXT_COLUMN_PREFIX + c if c != key else c for c in df_nxt.columns]

    # merge all together
    df = pd.merge(df_ctx, df_cnt, on=key, how="left")
    df = pd.merge(df, df_tgt, on=key, how="left")
    df = pd.merge(df, df_nxt, on=key, how="left")
    df = df.drop(columns=[key])

    # remove records with sequence length equal to 0
    count_column = f"{TGT_COLUMN_PREFIX}{COUNT_COLUMN}"
    df[count_column] = df[count_column].fillna(0).astype("Int64")
    df = df.loc[df[count_column] > 0].reset_index(drop=True)

    if setup is None:
        setup = "1:1" if (df[count_column] == 1).all() else "1:N"

    # for 1:1 ctx/tgt setups, drop nxt and count columns; ensure at least one column remains
    if setup == "1:1":
        df = df.drop(columns=[c for c in df.columns if c.startswith(NXT_COLUMN_PREFIX)])
    if setup == "1:1" and len(df.columns) > 1:
        df = df.drop(columns=[count_column])

    # harmonize dtypes
    df = df.apply(harmonize_dtype)

    # sample tokens from text-like columns
    df = sample_text_tokens(df)

    return df.reset_index(drop=True)


def sample_two_consecutive_rows(
    df: pd.DataFrame,
    col_by: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples two consecutive rows for each group in a DataFrame.

    If a group has only one row, the second row will be missing.
    """

    # enrich data with index column
    df["__IDX"] = df.groupby(col_by).cumcount()

    # determine sequence lengths for each group
    seq_lens = df.groupby(col_by).size()

    # make random draw from [0, seq_len-1]
    sel_idx = (seq_lens - 1) * np.random.random(len(seq_lens)).astype("int")
    sel_idx_df = pd.Series(sel_idx).to_frame("__IDX").reset_index()

    # filter to randomly selected indices
    first_rows = df.merge(sel_idx_df, on=[col_by, "__IDX"])

    # filter to succeeding rows of selected indices
    sel_idx_df["__IDX"] += 1
    second_rows = df.merge(sel_idx_df, on=[col_by, "__IDX"])

    # drop temporary index columns
    first_rows.drop(columns=["__IDX"], inplace=True)
    second_rows.drop(columns=["__IDX"], inplace=True)

    return first_rows, second_rows


def pull_data_for_embeddings(
    *,
    df_tgt: pd.DataFrame,
    df_ctx: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    max_sample_size: int | None = None,
) -> pd.Series:
    t0 = time.time()

    # keys must be provided if df_ctx provided
    assert df_ctx is None or (ctx_primary_key is not None and tgt_context_key is not None)

    # sort columns to ensure consistent column order
    df_tgt = df_tgt[sorted(df_tgt.columns)]
    if df_ctx is not None:
        df_ctx = df_ctx[sorted(df_ctx.columns)]

    key = "__KEY"

    if df_ctx is not None:
        # explicit context
        df_ctx = df_ctx.sample(frac=1).head(max_sample_size)
        df_ctx = df_ctx.rename(columns={ctx_primary_key: tgt_context_key}).reset_index(drop=True)
        df_tgt = df_tgt.merge(df_ctx[tgt_context_key], on=tgt_context_key, how="right").reset_index(drop=True)
    elif tgt_context_key is not None:
        # implicit context
        df_ctx = df_tgt[[tgt_context_key]].drop_duplicates()
        df_ctx = df_ctx.sample(frac=1).head(max_sample_size).reset_index(drop=True)
        df_tgt = df_tgt.merge(df_ctx[tgt_context_key], on=tgt_context_key, how="right").reset_index(drop=True)
    else:
        # no context; flat table
        tgt_context_key = key
        df_tgt = df_tgt.sample(frac=1).head(max_sample_size).reset_index(drop=True)
        df_tgt[tgt_context_key] = range(len(df_tgt))

    # consistently use "__KEY" as key column
    df_tgt = df_tgt.rename(columns={tgt_context_key: key})
    tgt_context_key = key

    # harmonize numerical columns to double precision
    num_cols = df_tgt.select_dtypes("number").columns.drop(tgt_context_key, errors="ignore")
    df_tgt[num_cols] = df_tgt[num_cols].astype("float64[pyarrow]")

    def row_to_string(row: pd.Series) -> str:
        # we concatenate all values as strings rather than convert to
        # JSON to keep the string length for faster speed short
        return " ".join(row.values.astype(str))

    def sequence_to_string(sequence: pd.DataFrame) -> str:
        return ", ".join(sequence.apply(row_to_string, axis=1))

    strings = (
        df_tgt.groupby(tgt_context_key)
        .apply(sequence_to_string, include_groups=False)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    time_elapsed = time.time() - t0
    _LOG.info(f"finished pulling data for embeddings ({time_elapsed=:.2f}s, {strings.shape=})")
    return strings


def calculate_embeddings(texts: pd.Series | pd.DataFrame) -> np.ndarray:
    t0 = time.time()
    if isinstance(texts, pd.DataFrame):
        # convert to JSON strings
        texts = pd.Series(texts.to_dict(orient="records"))
    # cap at 1k chars, as encoder truncates anyway; still it speeds things up by truncating beforehand
    texts = texts.astype("string[pyarrow]").str[:1_000]
    # calculate embeddings for text column
    encoder = load_embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = encoder.encode(texts.tolist())
    time_elapsed = time.time() - t0
    _LOG.info(f"created embeddings for {len(texts):,} records ({time_elapsed=:.2f}s)")
    return embeddings


def sample_text_tokens(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = load_tokenizer()

    def tokenize_and_sample(text: str | None) -> str | None:
        if pd.isna(text) or text == "":
            return None
        tokens = tokenizer.tokenize(text)
        tokens = (t.replace("Ġ", "▁") for t in tokens)  # replace initial space with thick underscore
        return random.choice(list(tokens))

    def process_text_columns(x: pd.Series) -> pd.Series:
        if not is_text_heuristic(x):
            return x
        return x.apply(tokenize_and_sample)

    return df.apply(process_text_columns)


def harmonize_dtype(x: pd.Series):
    # Convert to a small set of nullable dtypes, so that we avoid issues if
    # there is a dtype mismatch between `tgt` and `syn`. We leave dtype
    # as-is in case of casting error, to continue QA.

    def is_timestamp_dtype(x: pd.Series) -> bool:
        if isinstance(x.dtype, pd.ArrowDtype):
            return pa.types.is_timestamp(x.dtype.pyarrow_dtype)
        else:
            return pd.api.types.is_datetime64_any_dtype(x)

    try:
        if is_timestamp_dtype(x):
            x = x.astype("datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(x):
            x = x.astype("Float64")
        else:
            x = x.astype("object")
    except Exception:
        # leave dtype as-is, but just log a warning message
        pass
    return x


def is_text_heuristic(x: pd.Series) -> bool:
    # if more than 5% of rows contain unique values -> consider as TEXT
    return x.dtype == "object" and x.value_counts().eq(1).reindex(x).mean() > 0.05
