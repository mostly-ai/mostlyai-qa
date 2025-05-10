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


def preprocess_numerics(trn: pd.DataFrame, syn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trn_num, syn_num = {}, {}
    for col in trn.columns:
        # convert to numerics
        trn_num[col] = pd.to_numeric(trn[col], errors="coerce")
        syn_num[col] = pd.to_numeric(syn[col], errors="coerce")
        # retain NAs (needed for datetime)
        trn_num[col] = trn_num[col].where(~trn[col].isna(), np.nan)
        syn_num[col] = syn_num[col].where(~syn[col].isna(), np.nan)
        # normalize numeric features based on trn
        qt_scaler = QuantileTransformer(
            output_distribution="uniform",
            random_state=42,
            n_quantiles=min(100, len(trn)),
        )
        qt_scaler.fit(trn_num[col].values.reshape(-1, 1))
        trn_num[col] = qt_scaler.transform(trn_num[col].values.reshape(-1, 1))[:, 0]
        syn_num[col] = qt_scaler.transform(syn_num[col].values.reshape(-1, 1))[:, 0]
        # replace NAs with 0.5
        trn_num[col] = np.nan_to_num(trn_num[col], nan=0.5)
        syn_num[col] = np.nan_to_num(syn_num[col], nan=0.5)
        # add extra columns for NAs
        trn_num[col + " - N/A"] = trn[col].isna().astype(float)
        syn_num[col + " - N/A"] = syn[col].isna().astype(float)
    return pd.DataFrame(trn_num), pd.DataFrame(syn_num)


def preprocess_categoricals(trn: pd.DataFrame, syn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trn_cat, syn_cat = {}, {}
    for col in trn.columns:
        # prepare inputs
        trn_col = trn[col].astype(str).fillna(NA_BIN).replace("", EMPTY_BIN)
        syn_col = syn[col].astype(str).fillna(NA_BIN).replace("", EMPTY_BIN)
        syn_col = syn_col.where(syn_col.isin(trn_col.values), RARE_BIN)
        # get unique values
        uvals = trn_col.value_counts().index.to_list() + [RARE_BIN]
        # embed unique values into high-dimensional space
        from mostlyai.qa.assets import load_embedder

        embedder = load_embedder()
        embeds = embedder.encode(uvals)
        # project embeddings into a low-dimensional space
        from sklearn.decomposition import PCA

        dims = 2  # potentially adapt to the number of unique values
        pca = PCA(n_components=dims)
        embeds = pca.fit_transform(embeds)
        # create mapping from unique values to PCA
        embeds = pd.DataFrame(embeds)
        embeds.index = uvals
        # map values to PCA
        trn_cat[col] = embeds.reindex(trn_col.values).reset_index(drop=True)
        syn_cat[col] = embeds.reindex(syn_col.values).reset_index(drop=True)
        # assign column names
        trn_cat[col].columns = [f"{col} - PCA {i + 1}" for i in range(dims)]
        syn_cat[col].columns = [f"{col} - PCA {i + 1}" for i in range(dims)]
    trn_cat = pd.concat(trn_cat.values(), axis=1)
    syn_cat = pd.concat(syn_cat.values(), axis=1)
    return trn_cat, syn_cat
