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
import time

import networkx as nx
import numpy as np
from joblib import cpu_count
from plotly import graph_objs as go
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors

from mostlyai.qa._common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
)
from mostlyai.qa._filesystem import TemporaryWorkspace

_LOG = logging.getLogger(__name__)


def calculate_dcrs_nndrs(
    data: np.ndarray | None, query: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculate Distance to Closest Records (DCRs) and Nearest Neighbor Distance Ratios (NNDRs).
    """
    if data is None or query is None or data.shape[0] == 0 or query.shape[0] == 0:
        return None, None
    t0 = time.time()
    data = data[data[:, 0].argsort()]  # sort data by first dimension to enforce deterministic results

    index = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="l2", n_jobs=min(16, max(1, cpu_count() - 1)))
    index.fit(data)
    dcrs, _ = index.kneighbors(query)
    dcr = dcrs[:, 0]
    nndr = (dcrs[:, 0] + 1e-8) / (dcrs[:, 1] + 1e-8)
    _LOG.info(f"calculated DCRs for {data.shape=} and {query.shape=} in {time.time() - t0:.2f}s")
    return dcr, nndr


def calculate_distances(
    *, syn_embeds: np.ndarray, trn_embeds: np.ndarray, hol_embeds: np.ndarray | None
) -> dict[str, np.ndarray]:
    """
    Calculates distances to the closest records (DCR).
    """
    assert syn_embeds.shape == trn_embeds.shape
    if hol_embeds is not None and hol_embeds.shape[0] > 0:
        assert trn_embeds.shape == hol_embeds.shape

    if hol_embeds is None:
        # calculate DCR / NNDR for synthetic to training
        dcr_syn_trn, nndr_syn_trn = calculate_dcrs_nndrs(data=trn_embeds, query=syn_embeds)
        dcr_syn_hol, nndr_syn_hol = None, None
        dcr_trn_hol, nndr_trn_hol = None, None
    else:
        # calculate DCR / NNDR for several (sub)sets of columns and keep the one with highest DCR share
        ori_embeds = np.vstack((trn_embeds, hol_embeds))
        groups = []
        # check all columns together
        groups += [np.arange(ori_embeds.shape[1])]
        # check 3 correlated subsets of columns
        if ori_embeds.shape[1] > 10:
            groups += split_columns_into_correlated_groups(ori_embeds, k=3)
        # check 3 random subsets of columns
        if ori_embeds.shape[1] > 10:
            groups += split_columns_into_random_groups(ori_embeds, k=3)
        dcr_share = 0.0
        nndr_ratio = 1.0
        for columns in groups:
            # calculate DCR / NNDR for synthetic to training
            g_dcr_syn_trn, g_nndr_syn_trn = calculate_dcrs_nndrs(
                data=trn_embeds[:, columns], query=syn_embeds[:, columns]
            )
            # calculate DCR / NNDR for synthetic to holdout
            g_dcr_syn_hol, g_nndr_syn_hol = calculate_dcrs_nndrs(
                data=hol_embeds[:, columns], query=syn_embeds[:, columns]
            )
            # calculate DCR / NNDR for holdout to training
            g_dcr_trn_hol, g_nndr_trn_hol = calculate_dcrs_nndrs(
                data=trn_embeds[:, columns], query=hol_embeds[:, columns]
            )
            # keep results if DCR share is MAX
            g_dcr_share = calculate_dcr_share(g_dcr_syn_trn, g_dcr_syn_hol)
            g_nndr_ratio = calculate_nndr_ratio(g_nndr_syn_trn, g_nndr_syn_hol)
            if len(columns) == ori_embeds.shape[1]:
                suffix = "ALL columns"
            else:
                suffix = f"{len(columns)} columns [{columns}]"
            _LOG.info(f"DCR Share: {g_dcr_share:.1%}, NNDR Ratio: {g_nndr_ratio:.3f} - {suffix}")
            if g_dcr_share >= dcr_share:
                # keep results if DCR share is MAX
                dcr_share = g_dcr_share
                nndr_ratio = g_nndr_ratio
                dcr_syn_trn, nndr_syn_trn = g_dcr_syn_trn, g_nndr_syn_trn
                dcr_syn_hol, nndr_syn_hol = g_dcr_syn_hol, g_nndr_syn_hol
                dcr_trn_hol, nndr_trn_hol = g_dcr_trn_hol, g_nndr_trn_hol
        _LOG.info(f"DCR Share: {dcr_share:.1%}, NNDR Ratio: {nndr_ratio:.3f} - FINAL")

    return {
        "dcr_syn_trn": dcr_syn_trn,
        "nndr_syn_trn": nndr_syn_trn,
        "dcr_syn_hol": dcr_syn_hol,
        "nndr_syn_hol": nndr_syn_hol,
        "dcr_trn_hol": dcr_trn_hol,
        "nndr_trn_hol": nndr_trn_hol,
    }


def deciles(x):
    return np.round(np.quantile(x, np.linspace(0, 1, 11)), 3)


def calculate_ims(dcr: np.ndarray) -> float:
    return (dcr <= 1e-6).mean()


def calculate_dcr(dcr: np.ndarray) -> float:
    return dcr.mean()


def calculate_dcr_share(dcr_syn_trn: np.ndarray, dcr_syn_hol: np.ndarray) -> float:
    return np.mean(dcr_syn_trn < dcr_syn_hol) + np.mean(dcr_syn_trn == dcr_syn_hol) / 2


def calculate_nndr(nndrs: np.ndarray) -> float:
    return np.sort(nndrs)[9]


def calculate_nndr_ratio(nndr_syn_trn: np.ndarray, nndr_syn_hol: np.ndarray) -> float:
    return calculate_nndr(nndr_syn_trn) / calculate_nndr(nndr_syn_hol)


def split_columns_into_random_groups(X, k):
    """
    Split the columns of input matrix X into k non-overlapping, randomly ordered groups
    with as even sizes as possible (difference ≤ 1).

    Parameters:
        X (ndarray): Input array of shape (n_samples, n_features)
        k (int): Number of groups to split columns into

    Returns:
        List of lists: Each list contains column indices for one group
    """
    n_cols = X.shape[1]

    # shuffle all column indices
    all_indices = np.arange(n_cols)
    np.random.shuffle(all_indices)

    # evenly divide shuffled indices into k groups
    base_size = n_cols // k
    remainder = n_cols % k
    groups = []

    start = 0
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        groups.append(all_indices[start : start + size].tolist())
        start += size

    return groups


def split_columns_into_correlated_groups(X, k):
    """
    Split the columns of input matrix X into k groups such that
    intra-group correlation is high and cross-group correlation is minimized.
    Uses spectral clustering on a correlation-weighted graph.

    Parameters:
        X (ndarray): Input data of shape (n_samples, n_features)
        k (int): Number of desired column groups

    Returns:
        groups (list of lists): List containing k lists of column indices
    """

    def correlation_graph(X):
        """
        Constructs a graph where each node is a feature (column),
        and edges are weighted by absolute Pearson correlation.

        Returns:
            G (networkx.Graph): Weighted undirected graph
            corr_matrix (ndarray): Absolute correlation matrix
        """
        n = X.shape[1]
        # add tiny noise to avoid zero variance
        X_noisy = X + 1e-8 * np.random.randn(*X.shape)
        corr_matrix = np.abs(np.corrcoef(X_noisy, rowvar=False))
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=corr_matrix[i, j])
        return G, corr_matrix

    # Step 1: Create correlation graph and matrix
    G, _ = correlation_graph(X)

    # Step 2: Convert graph to adjacency matrix (symmetric)
    adj_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i, j, d in G.edges(data=True):
        adj_matrix[i, j] = d["weight"]
        adj_matrix[j, i] = d["weight"]

    # Step 3: Apply spectral clustering to partition the graph
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",  # uses adj_matrix directly as similarity
        assign_labels="kmeans",  # clustering on the embedding
    )
    try:
        labels = sc.fit_predict(adj_matrix)
    except Exception as e:
        _LOG.warning(f"Spectral clustering failed: {e}")
        return []

    # Step 4: Group column indices by their cluster labels
    groups = [np.where(labels == i)[0].tolist() for i in range(k)]
    return groups


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
