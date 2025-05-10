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
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from mostlyai.qa._common import EMPTY_BIN, NA_BIN, RARE_BIN
from mostlyai.qa.assets import load_embedder
from sklearn.decomposition import PCA


def detect_feature_types(df, cat_unique_thresh=20):
    feature_types = {}
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            feature_types[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) <= cat_unique_thresh:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numeric"
        else:
            feature_types[col] = "categorical"
    return feature_types


def encode_numerics(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        qt_scaler.fit(pd.concat([trn_num[col], hol_num[col]]).values.reshape(-1, 1))
        syn_num[col] = qt_scaler.transform(syn_num[col].values.reshape(-1, 1))[:, 0]
        trn_num[col] = qt_scaler.transform(trn_num[col].values.reshape(-1, 1))[:, 0]
        hol_num[col] = qt_scaler.transform(hol_num[col].values.reshape(-1, 1))[:, 0]
        # replace NAs with 0.5
        syn_num[col] = np.nan_to_num(syn_num[col], nan=0.5)
        trn_num[col] = np.nan_to_num(trn_num[col], nan=0.5)
        hol_num[col] = np.nan_to_num(hol_num[col], nan=0.5)
        # add extra columns for NAs
        syn_num[col + " - N/A"] = syn[col].isna().astype(float)
        trn_num[col + " - N/A"] = trn[col].isna().astype(float)
        hol_num[col + " - N/A"] = hol[col].isna().astype(float)
    return pd.DataFrame(syn_num), pd.DataFrame(trn_num), pd.DataFrame(hol_num)


def encode_categoricals(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trn_cat, syn_cat, hol_cat = {}, {}, {}
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
        # project embeddings into a low-dimensional space
        dims = 2  # potentially adapt to the number of unique values
        pca = PCA(n_components=dims)
        embeds = pca.fit_transform(embeds)
        # create mapping from unique values to PCA
        embeds = pd.DataFrame(embeds)
        embeds.index = uvals + [RARE_BIN]
        # map values to PCA
        syn_cat[col] = embeds.reindex(syn_col.values).reset_index(drop=True)
        trn_cat[col] = embeds.reindex(trn_col.values).reset_index(drop=True)
        hol_cat[col] = embeds.reindex(hol_col.values).reset_index(drop=True)
        # assign column names
        columns = [f"{col} - PCA {i + 1}" for i in range(dims)]
        syn_cat[col].columns = columns
        trn_cat[col].columns = columns
        hol_cat[col].columns = columns
    syn_cat = pd.concat(syn_cat.values(), axis=1)
    trn_cat = pd.concat(trn_cat.values(), axis=1)
    hol_cat = pd.concat(hol_cat.values(), axis=1)
    return syn_cat, trn_cat, hol_cat


def encode(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_dat_cols = trn.select_dtypes(include=["number", "datetime"]).columns
    other_cols = [col for col in trn.columns if col not in num_dat_cols]
    trn_num, syn_num, hol_num = encode_numerics(trn[num_dat_cols], syn[num_dat_cols], hol[num_dat_cols])
    trn_cat, syn_cat, hol_cat = encode_categoricals(trn[other_cols], syn[other_cols], hol[other_cols])
    syn_encoded = pd.concat([syn_num, syn_cat], axis=1)
    trn_encoded = pd.concat([trn_num, trn_cat], axis=1)
    hol_encoded = pd.concat([hol_num, hol_cat], axis=1)
    return syn_encoded, trn_encoded, hol_encoded
