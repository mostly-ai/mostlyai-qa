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
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.common import is_datetime64_dtype, is_numeric_dtype

from mostlyai.qa._accuracy import bin_data
from mostlyai.qa._common import (
    ACCURACY_MAX_COLUMNS,
    COUNT_COLUMN,
    CTX_COLUMN_PREFIX,
    EMBEDDINGS_MAX_COLUMNS,
    NXT_COLUMN_PREFIX,
    TGT_COLUMN_PREFIX,
)
from mostlyai.qa._embeddings import encode_data
from mostlyai.qa.assets import load_tokenizer

_LOG = logging.getLogger(__name__)


def prepare_data_for_accuracy(
    *,
    df_tgt: pd.DataFrame,
    df_ctx: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    max_sample_size: int | None = None,
    setup: str | None = None,
    ori_dtypes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Prepare data for accuracy plots and metrics.
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
    count_column = f"{TGT_COLUMN_PREFIX}{COUNT_COLUMN}"
    df[count_column] = df[count_column].fillna(0).astype("Int64")

    # determine setup if not provided
    if setup is None:
        setup = "1:1" if (df[count_column] == 1).all() else "1:N"

    # remove records with sequence length equal to 0
    df = df.loc[df[count_column] > 0].reset_index(drop=True)

    # for 1:1 ctx/tgt setups, drop nxt and count columns; ensure at least one column remains
    if setup == "1:1":
        df = df.drop(columns=[c for c in df.columns if c.startswith(NXT_COLUMN_PREFIX)])
    if setup == "1:1" and len(df.columns) > 1:
        df = df.drop(columns=[count_column])

    # harmonize dtypes
    df = df.apply(harmonize_dtype)

    # coerce dtypes to ori_dtypes
    for trn_col, trn_dtype in (ori_dtypes or {}).items():
        if is_numeric_dtype(trn_dtype):
            df[trn_col] = pd.to_numeric(df[trn_col], errors="coerce")
        elif is_datetime64_dtype(trn_dtype):
            df[trn_col] = pd.to_datetime(df[trn_col], errors="coerce")
        df[trn_col] = df[trn_col].astype(trn_dtype)

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
    sel_idx = ((seq_lens - 1) * np.random.random(len(seq_lens))).astype("int")
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


def prepare_data_for_coherence(
    *,
    df_tgt: pd.DataFrame,
    tgt_context_key: str,
    bins: int | dict[str, list] = 30,
    max_sequence_length: int = 100,
    max_sample_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    df_tgt = df_tgt.copy()

    # limit sample size to at most max_sample_size sequences
    keys = df_tgt[tgt_context_key].drop_duplicates().sample(frac=1).head(max_sample_size)
    df_tgt = df_tgt[df_tgt[tgt_context_key].isin(keys)].reset_index(drop=True)

    # randomly sample at most max_sequence_length rows per sequence
    df_tgt = df_tgt.sample(frac=1).reset_index(drop=True)
    df_tgt = df_tgt[df_tgt.groupby(tgt_context_key).cumcount() < max_sequence_length].reset_index(drop=True)

    # split into key and non-key columns
    non_key_cols = [c for c in df_tgt.columns if c != tgt_context_key]
    keys = df_tgt[tgt_context_key]
    df_tgt = df_tgt[non_key_cols]

    # apply harmonize_dtype to all columns except tgt_context_key
    df_tgt = df_tgt.apply(harmonize_dtype)

    # sample tokens from text-like columns except tgt_context_key
    df_tgt = sample_text_tokens(df_tgt)

    # discretize all columns except tgt_context_key
    df_tgt, bins = bin_data(df_tgt, bins=bins, non_categorical_label_style="bounded")

    # merge keys with binned data
    df_tgt = pd.concat([keys, df_tgt], axis=1)

    return df_tgt, bins


def sample_text_tokens(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = load_tokenizer()

    def tokenize_and_sample(text: Any) -> str | None:
        if pd.isna(text) or text == "":
            return None
        tokens = tokenizer.tokenize(str(text))
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
        # leave dtype as-is
        pass
    return x


def is_text_heuristic(x: pd.Series) -> bool:
    # if more than 5% of rows contain unique values -> consider as TEXT
    return x.dtype == "object" and x.value_counts().eq(1).reindex(x).mean() > 0.05


def prepare_data_for_embeddings(
    *,
    syn_tgt_data: pd.DataFrame,
    trn_tgt_data: pd.DataFrame,
    hol_tgt_data: pd.DataFrame | None = None,
    syn_ctx_data: pd.DataFrame | None = None,
    trn_ctx_data: pd.DataFrame | None = None,
    hol_ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    max_sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # helper variables
    key = tgt_context_key or None
    hol = hol_tgt_data is not None

    # filter target to context keys
    if trn_ctx_data is not None:
        rename_key = {ctx_primary_key: key}
        syn_ctx_data = syn_ctx_data[[ctx_primary_key]].rename(columns=rename_key)
        trn_ctx_data = trn_ctx_data[[ctx_primary_key]].rename(columns=rename_key)
        hol_ctx_data = hol_ctx_data[[ctx_primary_key]].rename(columns=rename_key) if hol else None
        syn_tgt_data = syn_tgt_data.merge(syn_ctx_data, on=key, how="inner")
        trn_tgt_data = trn_tgt_data.merge(trn_ctx_data, on=key, how="inner")
        hol_tgt_data = hol_tgt_data.merge(hol_ctx_data, on=key, how="inner") if hol else None

    # enrich with count column
    if tgt_context_key is not None:
        syn_tgt_data.insert(0, COUNT_COLUMN, syn_tgt_data.groupby(key)[key].transform("size"))
        trn_tgt_data.insert(0, COUNT_COLUMN, trn_tgt_data.groupby(key)[key].transform("size"))
        hol_tgt_data.insert(0, COUNT_COLUMN, hol_tgt_data.groupby(key)[key].transform("size")) if hol else None

    # cap to Q95 sequence length of original to avoid excessive samples per group distorting results
    if tgt_context_key is not None:
        cap_sequence_length = 100
        q95_sequence_length = trn_tgt_data.groupby(key).size().quantile(0.95)
        max_sequence_length = min(q95_sequence_length, cap_sequence_length)
        syn_tgt_data = syn_tgt_data.groupby(key).sample(frac=1).groupby(key).head(n=max_sequence_length)
        trn_tgt_data = trn_tgt_data.groupby(key).sample(frac=1).groupby(key).head(n=max_sequence_length)
        hol_tgt_data = (
            hol_tgt_data.groupby(key).sample(frac=1).groupby(key).head(n=max_sequence_length) if hol else None
        )

    # drop key from data as its not relevant for embeddings
    if tgt_context_key is not None:
        syn_tgt_data = syn_tgt_data.drop(columns=[key])
        trn_tgt_data = trn_tgt_data.drop(columns=[key])
        hol_tgt_data = hol_tgt_data.drop(columns=[key]) if hol else None

    # draw equally sized samples for fair 3-way comparison
    max_sample_size_final = min(
        max_sample_size or float("inf"),
        len(syn_tgt_data),
        len(trn_tgt_data),
        len(hol_tgt_data) if hol_tgt_data is not None else float("inf"),
    )
    syn_tgt_data = syn_tgt_data.sample(n=max_sample_size_final)
    trn_tgt_data = trn_tgt_data.sample(n=max_sample_size_final)
    hol_tgt_data = hol_tgt_data.sample(n=max_sample_size_final) if hol else None

    if max_sample_size_final > 50_000 and max_sample_size is None:
        warnings.warn(
            UserWarning(
                "More than 50k embeddings will be calculated per dataset, which may take a long time. "
                "Consider setting a limit via `max_sample_size_embeddings` to speed up the process. "
                "Note however, that limiting the number of embeddings will affect the sensitivity of the distance metrics."
            )
        )

    # limit to same columns
    trn_cols = list(trn_tgt_data.columns)[:EMBEDDINGS_MAX_COLUMNS]
    syn_tgt_data = syn_tgt_data[trn_cols]
    trn_tgt_data = trn_tgt_data[trn_cols]
    hol_tgt_data = hol_tgt_data[trn_cols] if hol else None

    # harmonize dtypes
    syn_tgt_data = syn_tgt_data.apply(harmonize_dtype)
    trn_tgt_data = trn_tgt_data.apply(harmonize_dtype)
    hol_tgt_data = hol_tgt_data.apply(harmonize_dtype) if hol else None

    # encode data
    syn_embeds, trn_embeds, hol_embeds = encode_data(
        syn_data=syn_tgt_data,
        trn_data=trn_tgt_data,
        hol_data=hol_tgt_data,
    )

    return syn_embeds, trn_embeds, hol_embeds
