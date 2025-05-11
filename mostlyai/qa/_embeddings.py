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

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize

from mostlyai.qa._common import (
    COUNT_COLUMN,
    EMPTY_BIN,
    NA_BIN,
    NXT_COLUMN_PREFIX,
    RARE_BIN,
    TGT_COLUMN_PREFIX,
)
from mostlyai.qa.assets import load_embedder


_LOG = logging.getLogger(__name__)


def encode_numerics(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Encode numeric features by mapping this via QuantileTransformer to a uniform distribution from [-0.5, 0.5].
    """
    syn_num, trn_num, hol_num = {}, {}, {}
    if hol is None:
        hol = pd.DataFrame(columns=trn.columns)
    for col in trn.columns:
        # convert to numerics
        syn_num[col] = pd.to_numeric(syn[col], errors="coerce")
        trn_num[col] = pd.to_numeric(trn[col], errors="coerce")
        hol_num[col] = pd.to_numeric(hol[col], errors="coerce")
        # retain NAs (needed for datetime)
        syn_num[col] = syn_num[col].where(~syn[col].isna(), np.nan)
        trn_num[col] = trn_num[col].where(~trn[col].isna(), np.nan)
        hol_num[col] = hol_num[col].where(~hol[col].isna(), np.nan)
        # normalize numeric features based on trn
        qt_scaler = QuantileTransformer(
            output_distribution="uniform",
            random_state=42,
            n_quantiles=min(100, len(trn) + len(hol)),
        )
        ori_num = pd.concat([trn_num[col], hol_num[col]]) if len(hol) > 0 else pd.DataFrame(trn_num[col])
        qt_scaler.fit(ori_num.values.reshape(-1, 1))
        syn_num[col] = qt_scaler.transform(syn_num[col].values.reshape(-1, 1))[:, 0] - 0.5
        trn_num[col] = qt_scaler.transform(trn_num[col].values.reshape(-1, 1))[:, 0] - 0.5
        hol_num[col] = qt_scaler.transform(hol_num[col].values.reshape(-1, 1))[:, 0] - 0.5 if len(hol) > 0 else None
        # replace NAs with 0.0
        syn_num[col] = np.nan_to_num(syn_num[col], nan=0.0)
        trn_num[col] = np.nan_to_num(trn_num[col], nan=0.0)
        hol_num[col] = np.nan_to_num(hol_num[col], nan=0.0)
        # add extra columns for NAs
        if trn[col].isna().any() or hol[col].isna().any():
            syn_num[col + " - N/A"] = syn[col].isna().astype(float)
            trn_num[col + " - N/A"] = trn[col].isna().astype(float)
            hol_num[col + " - N/A"] = hol[col].isna().astype(float)
    syn_num = pd.DataFrame(syn_num, index=syn.index)
    trn_num = pd.DataFrame(trn_num, index=trn.index)
    hol_num = pd.DataFrame(hol_num, index=hol.index) if len(hol) > 0 else None
    return syn_num, trn_num, hol_num


def encode_strings(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Encode string features by mapping them to a low-dimensional space using PCA of their embeddings.
    """
    trn_str, syn_str, hol_str = {}, {}, {}
    if hol is None:
        hol = pd.DataFrame(columns=trn.columns)
    for col in trn.columns:
        # prepare inputs
        syn_col = syn[col].astype(str).fillna(NA_BIN).replace("", EMPTY_BIN)
        trn_col = trn[col].astype(str).fillna(NA_BIN).replace("", EMPTY_BIN)
        hol_col = hol[col].astype(str).fillna(NA_BIN).replace("", EMPTY_BIN)
        # get unique original values
        uvals = pd.concat([trn_col, hol_col]).value_counts().index.to_list()
        # map out of range values to RARE_BIN
        syn_col = syn_col.where(syn_col.isin(uvals), RARE_BIN)
        # embed unique values into high-dimensional space
        embedder = load_embedder()
        embeds = embedder.encode(uvals + [RARE_BIN])
        # project embeddings into a low-dimensional space with dim depending on col cardinality
        if len(uvals) <= 20:
            dims = 2
        elif len(uvals) <= 100:
            dims = 3
        else:
            dims = 4
        pca_model = PCA(n_components=dims)
        embeds = pca_model.fit_transform(embeds)
        # create mapping from unique values to PCA
        embeds = pd.DataFrame(embeds)
        embeds.index = uvals + [RARE_BIN]
        # map values to PCA
        syn_str[col] = embeds.reindex(syn_col.values).reset_index(drop=True)
        trn_str[col] = embeds.reindex(trn_col.values).reset_index(drop=True)
        hol_str[col] = embeds.reindex(hol_col.values).reset_index(drop=True)
        # assign column names
        columns = [f"{col} - PCA {i + 1}" for i in range(dims)]
        syn_str[col].columns = columns
        trn_str[col].columns = columns
        hol_str[col].columns = columns
    syn_str = pd.concat(syn_str.values(), axis=1) if syn_str else pd.DataFrame()
    syn_str.index = syn.index
    trn_str = pd.concat(trn_str.values(), axis=1) if trn_str else pd.DataFrame()
    trn_str.index = trn.index
    if len(hol) > 0:
        hol_str = pd.concat(hol_str.values(), axis=1) if hol_str else pd.DataFrame()
        hol_str.index = hol.index
    else:
        hol_str = None
    return syn_str, trn_str, hol_str


def encode_data(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Encode all columns corresponding to their data type.
    """
    tgt_cols = [c for c in trn.columns if c.startswith(TGT_COLUMN_PREFIX)]
    nxt_cols = [c for c in trn.columns if c.startswith(NXT_COLUMN_PREFIX)]
    cnt_col = f"{TGT_COLUMN_PREFIX}{COUNT_COLUMN}"
    cnt_cols = [cnt_col] if cnt_col in trn.columns else []
    # split into numeric and string columns
    num_dat_cols = [
        col for col in tgt_cols if pd.api.types.is_numeric_dtype(trn[col]) or pd.api.types.is_datetime64_dtype(trn[col])
    ]
    string_cols = [col for col in tgt_cols if col not in num_dat_cols]
    # keep TGT data
    syn_data = syn[tgt_cols]
    trn_data = trn[tgt_cols]
    hol_data = hol[tgt_cols] if hol is not None else None
    # append NXT data with TGT data to increase data coverage
    if len(nxt_cols) > 0:
        syn_nxt = syn[cnt_cols + nxt_cols]
        syn_nxt.columns = syn_nxt.columns.str.replace(NXT_COLUMN_PREFIX, TGT_COLUMN_PREFIX)
        syn_data = pd.concat([syn_data, syn_nxt], axis=0)
        trn_nxt = trn[cnt_cols + nxt_cols]
        trn_nxt.columns = trn_nxt.columns.str.replace(NXT_COLUMN_PREFIX, TGT_COLUMN_PREFIX)
        trn_data = pd.concat([trn_data, trn_nxt], axis=0)
        if hol is not None:
            hol_nxt = hol[cnt_cols + nxt_cols]
            hol_nxt.columns = hol_nxt.columns.str.replace(NXT_COLUMN_PREFIX, TGT_COLUMN_PREFIX)
            hol_data = pd.concat([hol_data, hol_nxt], axis=0)
    # encode numeric columns
    syn_num, trn_num, hol_num = encode_numerics(
        syn_data[num_dat_cols], trn_data[num_dat_cols], hol_data[num_dat_cols] if hol_data is not None else None
    )
    # encode string columns
    syn_str, trn_str, hol_str = encode_strings(
        syn_data[string_cols], trn_data[string_cols], hol_data[string_cols] if hol_data is not None else None
    )
    # concatenate numeric and string encoded columns
    syn_encoded = pd.concat([syn_num, syn_str], axis=1)
    trn_encoded = pd.concat([trn_num, trn_str], axis=1)
    hol_encoded = pd.concat([hol_num, hol_str], axis=1) if hol_data is not None else None
    # normalize embeddings
    syn_encoded = normalize(syn_encoded.values, norm="l2")
    trn_encoded = normalize(trn_encoded.values, norm="l2")
    hol_encoded = normalize(hol_encoded.values, norm="l2") if hol_encoded is not None else None
    return syn_encoded, trn_encoded, hol_encoded
