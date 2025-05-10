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

import logging
import platform
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from mostlyai.qa._common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
    EMPTY_BIN,
    NA_BIN,
    RARE_BIN,
)
from mostlyai.qa._filesystem import TemporaryWorkspace
from plotly import graph_objs as go

from mostlyai.qa.assets import load_embedder
from sklearn.decomposition import PCA

_LOG = logging.getLogger(__name__)


def encode_numerics(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
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
        syn_num[col] = qt_scaler.transform(syn_num[col].values.reshape(-1, 1))[:, 0]
        trn_num[col] = qt_scaler.transform(trn_num[col].values.reshape(-1, 1))[:, 0]
        hol_num[col] = qt_scaler.transform(hol_num[col].values.reshape(-1, 1))[:, 0] if len(hol) > 0 else None
        # replace NAs with 0.5
        syn_num[col] = np.nan_to_num(syn_num[col], nan=0.5)
        trn_num[col] = np.nan_to_num(trn_num[col], nan=0.5)
        hol_num[col] = np.nan_to_num(hol_num[col], nan=0.5)
        # add extra columns for NAs
        syn_num[col + " - N/A"] = syn[col].isna().astype(float)
        trn_num[col + " - N/A"] = trn[col].isna().astype(float)
        hol_num[col + " - N/A"] = hol[col].isna().astype(float)
    syn_num = pd.DataFrame(syn_num, index=syn.index)
    trn_num = pd.DataFrame(trn_num, index=trn.index)
    hol_num = pd.DataFrame(hol_num, index=hol.index) if len(hol) > 0 else None
    return syn_num, trn_num, hol_num


def encode_categoricals(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
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
        pca_model = PCA(n_components=dims)
        embeds = pca_model.fit_transform(embeds)
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
    syn_cat = pd.concat(syn_cat.values(), axis=1) if syn_cat else pd.DataFrame()
    syn_cat.index = syn.index
    trn_cat = pd.concat(trn_cat.values(), axis=1) if trn_cat else pd.DataFrame()
    trn_cat.index = trn.index
    if len(hol) > 0:
        hol_cat = pd.concat(hol_cat.values(), axis=1) if hol_cat else pd.DataFrame()
        hol_cat.index = hol.index
    else:
        hol_cat = None
    return syn_cat, trn_cat, hol_cat


def encode_data(
    syn: pd.DataFrame, trn: pd.DataFrame, hol: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    num_dat_cols = trn.select_dtypes(include=["number", "datetime"]).columns
    other_cols = [col for col in trn.columns if col not in num_dat_cols]
    syn_num, trn_num, hol_num = encode_numerics(
        syn[num_dat_cols], trn[num_dat_cols], hol[num_dat_cols] if hol is not None else None
    )
    syn_cat, trn_cat, hol_cat = encode_categoricals(
        syn[other_cols], trn[other_cols], hol[other_cols] if hol is not None else None
    )
    syn_encoded = pd.concat([syn_num, syn_cat], axis=1)
    trn_encoded = pd.concat([trn_num, trn_cat], axis=1)
    hol_encoded = pd.concat([hol_num, hol_cat], axis=1) if hol is not None else None
    return syn_encoded, trn_encoded, hol_encoded


def calculate_dcrs_nndrs(
    data: np.ndarray | None, query: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculate Distance to Closest Records (DCRs) and Nearest Neighbor Distance Ratios (NNDRs).

    Args:
        data: numeric base data
        query: numeric query data

    Returns:
        dcr: Distance to closest record
        nndr: Nearest neighbor distance ratio
    """
    if data is None or query is None or data.shape[0] == 0 or query.shape[0] == 0:
        return None, None
    _LOG.info(f"calculate DCRs for {data.shape=} and {query.shape=}")
    t0 = time.time()
    data = data[data[:, 0].argsort()]  # sort data by first dimension to enforce deterministic results
    if platform.system() == "Linux":
        # use FAISS on Linux for best performance
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(data.shape[1])  # inner product for cosine similarity with normalized vectors
        index.add(data)
        similarities, _ = index.search(query, 2)
        dcrs = np.clip(1 - similarities, 0, 1)
    else:
        # use sklearn as a fallback on non-Linux systems to avoid segfaults; these occurred when using QA as part of SDK
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        from joblib import cpu_count  # type: ignore

        index = NearestNeighbors(
            n_neighbors=2, algorithm="auto", metric="cosine", n_jobs=min(16, max(1, cpu_count() - 1))
        )
        index.fit(data)
        dcrs, _ = index.kneighbors(query)
    dcr = dcrs[:, 0]
    nndr = (dcrs[:, 0] + 1e-8) / (dcrs[:, 1] + 1e-8)
    _LOG.info(f"calculated DCRs for {data.shape=} and {query.shape=} in {time.time() - t0:.2f}s")
    return dcr, nndr


def calculate_distances(
    *, syn_encoded: np.ndarray, trn_encoded: np.ndarray, hol_encoded: np.ndarray | None
) -> dict[str, np.ndarray]:
    """
    Calculates distances to the closest records (DCR).

    Args:
        syn_encoded: Encoded synthetic data.
        trn_encoded: Encoded training data.
        hol_encoded: Encoded holdout data.

    Returns:
        Dictionary containing:
            - dcr_syn_trn: DCR for synthetic to training.
            - dcr_syn_hol: DCR for synthetic to holdout.
            - dcr_trn_hol: DCR for training to holdout.
            - nndr_syn_trn: NNDR for synthetic to training.
            - nndr_syn_hol: NNDR for synthetic to holdout.
            - nndr_trn_hol: NNDR for training to holdout.
    """
    assert syn_encoded.shape == trn_encoded.shape
    if hol_encoded is not None and hol_encoded.shape[0] > 0:
        assert trn_encoded.shape == hol_encoded.shape

    # calculate DCR / NNDR for synthetic to training
    dcr_syn_trn, nndr_syn_trn = calculate_dcrs_nndrs(data=trn_encoded, query=syn_encoded)
    # calculate DCR / NNDR for synthetic to holdout
    dcr_syn_hol, nndr_syn_hol = calculate_dcrs_nndrs(data=hol_encoded, query=syn_encoded)
    # calculate DCR / NNDR for holdout to training
    dcr_trn_hol, nndr_trn_hol = calculate_dcrs_nndrs(data=trn_encoded, query=hol_encoded)

    # log statistics
    def deciles(x):
        return np.round(np.quantile(x, np.linspace(0, 1, 11)), 3)

    _LOG.info(f"DCR deciles for synthetic to training: {deciles(dcr_syn_trn)}")
    _LOG.info(f"NNDR deciles for synthetic to training: {deciles(nndr_syn_trn)}")
    if dcr_syn_hol is not None:
        _LOG.info(f"DCR deciles for synthetic to holdout:  {deciles(dcr_syn_hol)}")
        _LOG.info(f"NNDR deciles for synthetic to holdout: {deciles(nndr_syn_hol)}")
        _LOG.info(f"share of dcr_syn_trn < dcr_syn_hol: {np.mean(dcr_syn_trn < dcr_syn_hol):.1%}")
        _LOG.info(f"share of nndr_syn_trn < nndr_syn_hol: {np.mean(nndr_syn_trn < nndr_syn_hol):.1%}")
        _LOG.info(f"share of dcr_syn_trn > dcr_syn_hol: {np.mean(dcr_syn_trn > dcr_syn_hol):.1%}")
        _LOG.info(f"share of nndr_syn_trn > nndr_syn_hol: {np.mean(nndr_syn_trn > nndr_syn_hol):.1%}")
    if dcr_trn_hol is not None:
        _LOG.info(f"DCR deciles for training to holdout:  {deciles(dcr_trn_hol)}")
        _LOG.info(f"NNDR deciles for training to holdout: {deciles(nndr_trn_hol)}")
    return {
        "dcr_syn_trn": dcr_syn_trn,
        "nndr_syn_trn": nndr_syn_trn,
        "dcr_syn_hol": dcr_syn_hol,
        "nndr_syn_hol": nndr_syn_hol,
        "dcr_trn_hol": dcr_trn_hol,
        "nndr_trn_hol": nndr_trn_hol,
    }


def plot_distances(plot_title: str, distances: dict[str, np.ndarray]) -> go.Figure:
    dcr_syn_trn = distances["dcr_syn_trn"]
    dcr_syn_hol = distances["dcr_syn_hol"]
    dcr_trn_hol = distances["dcr_trn_hol"]
    nndr_syn_trn = distances["nndr_syn_trn"]
    nndr_syn_hol = distances["nndr_syn_hol"]
    nndr_trn_hol = distances["nndr_trn_hol"]

    # calculate quantiles for DCR
    y = np.linspace(0, 1, 101)

    # Calculate max values to use later
    max_dcr_syn_trn = np.max(dcr_syn_trn)
    max_dcr_syn_hol = None if dcr_syn_hol is None else np.max(dcr_syn_hol)
    max_dcr_trn_hol = None if dcr_trn_hol is None else np.max(dcr_trn_hol)
    max_nndr_syn_trn = np.max(nndr_syn_trn)
    max_nndr_syn_hol = None if nndr_syn_hol is None else np.max(nndr_syn_hol)
    max_nndr_trn_hol = None if nndr_trn_hol is None else np.max(nndr_trn_hol)

    # Ensure first point is always at x=0 for all lines
    # and last point is at the maximum x value with y=1
    x_dcr_syn_trn = np.concatenate([[0], np.quantile(dcr_syn_trn, y[1:-1]), [max_dcr_syn_trn]])
    if dcr_syn_hol is not None:
        x_dcr_syn_hol = np.concatenate([[0], np.quantile(dcr_syn_hol, y[1:-1]), [max_dcr_syn_hol]])
    else:
        x_dcr_syn_hol = None

    if dcr_trn_hol is not None:
        x_dcr_trn_hol = np.concatenate([[0], np.quantile(dcr_trn_hol, y[1:-1]), [max_dcr_trn_hol]])
    else:
        x_dcr_trn_hol = None

    # calculate quantiles for NNDR
    x_nndr_syn_trn = np.concatenate([[0], np.quantile(nndr_syn_trn, y[1:-1]), [max_nndr_syn_trn]])
    if nndr_syn_hol is not None:
        x_nndr_syn_hol = np.concatenate([[0], np.quantile(nndr_syn_hol, y[1:-1]), [max_nndr_syn_hol]])
    else:
        x_nndr_syn_hol = None

    if nndr_trn_hol is not None:
        x_nndr_trn_hol = np.concatenate([[0], np.quantile(nndr_trn_hol, y[1:-1]), [max_nndr_trn_hol]])
    else:
        x_nndr_trn_hol = None

    # Adjust y to match the new x arrays with the added 0 and 1 points
    y = np.concatenate([[0], y[1:-1], [1]])

    # prepare layout
    layout = go.Layout(
        title=dict(text=f"<b>{plot_title}</b>", x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=dict(
            **CHARTS_FONTS["hover"],
            namelength=-1,  # Show full length of hover labels
        ),
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=500,
        margin=dict(l=20, r=20, b=20, t=60, pad=5),
        showlegend=True,
    )

    # Create a figure with two subplots side by side
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.05,
        subplot_titles=("Distance to Closest Record (DCR)", "Nearest Neighbor Distance Ratio (NNDR)"),
    )
    fig.update_annotations(font_size=12)

    # Configure axes for both subplots
    for i in range(1, 3):
        fig.update_xaxes(
            col=i,
            showline=True,
            linewidth=1,
            linecolor="#999999",
            hoverformat=".3f",
        )

        # Only show y-axis on the right side with percentage labels
        fig.update_yaxes(
            col=i,
            tickformat=".0%",
            showgrid=False,
            range=[-0.01, 1.01],
            showline=True,
            linewidth=1,
            linecolor="#999999",
            side="right",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

    # Add traces for DCR plot (left subplot)
    # training vs holdout (light gray)
    if x_dcr_trn_hol is not None:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=x_dcr_trn_hol,
                y=y,
                name="Training vs. Holdout Data",
                line=dict(color="#999999", width=5),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # synthetic vs holdout (gray)
    if x_dcr_syn_hol is not None:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=x_dcr_syn_hol,
                y=y,
                name="Synthetic vs. Holdout Data",
                line=dict(color="#666666", width=5),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # synthetic vs training (green)
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=x_dcr_syn_trn,
            y=y,
            name="Synthetic vs. Training Data",
            line=dict(color="#24db96", width=5),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add traces for NNDR plot (right subplot)
    # training vs holdout (light gray)
    if x_nndr_trn_hol is not None:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=x_nndr_trn_hol,
                y=y,
                name="Training vs. Holdout Data",
                line=dict(color="#999999", width=5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # synthetic vs holdout (gray)
    if x_nndr_syn_hol is not None:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=x_nndr_syn_hol,
                y=y,
                name="Synthetic vs. Holdout Data",
                line=dict(color="#666666", width=5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # synthetic vs training (green)
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=x_nndr_syn_trn,
            y=y,
            name="Synthetic vs. Training Data",
            line=dict(color="#24db96", width=5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            traceorder="reversed",
        )
    )

    return fig


def plot_store_distances(
    distances: dict[str, np.ndarray],
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_distances(
        "Cumulative Distributions of Distance Metrics",
        distances,
    )
    workspace.store_figure_html(fig, "distances_dcr")
