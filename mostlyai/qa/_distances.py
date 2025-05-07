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

import numpy as np

from mostlyai.qa._common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
)
from mostlyai.qa._filesystem import TemporaryWorkspace
from plotly import graph_objs as go
import faiss

_LOG = logging.getLogger(__name__)


def calculate_dcrs_nndrs(
    data: np.ndarray | None, query: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculate Distance to Closest Records (DCRs) and Nearest Neighbor Distance Ratios (NNDRs).

    Args:
        data: Embeddings of the training data.
        query: Embeddings of the query set.

    Returns:
    """
    if data is None or query is None:
        return None, None
    _LOG.info(f"calculate DCRs for {data.shape=} and {query.shape=}")
    t0 = time.time()
    data = data[data[:, 0].argsort()]  # sort data by first dimension to enforce deterministic results
    index = faiss.IndexFlatIP(data.shape[1])  # inner product for cosine similarity; embeddings are already normalized
    index.add(data)
    similarities, _ = index.search(query, 2)
    dcrs = np.clip(1 - similarities, 0, 1)
    dcr = dcrs[:, 0]
    nndr = (dcrs[:, 0] + 1e-8) / (dcrs[:, 1] + 1e-8)
    _LOG.info(f"calculated DCRs for {data.shape=} and {query.shape=} in {time.time() - t0:.2f}s")
    return dcr, nndr


def calculate_distances(
    *, syn_embeds: np.ndarray, trn_embeds: np.ndarray, hol_embeds: np.ndarray | None
) -> dict[str, np.ndarray]:
    """
    Calculates distances to the closest records (DCR).

    Args:
        syn_embeds: Embeddings of synthetic data.
        trn_embeds: Embeddings of training data.
        hol_embeds: Embeddings of holdout data.

    Returns:
        Dictionary containing:
            - dcr_syn_trn: DCR for synthetic to training.
            - dcr_syn_hol: DCR for synthetic to holdout.
            - dcr_trn_hol: DCR for training to holdout.
            - nndr_syn_trn: NNDR for synthetic to training.
            - nndr_syn_hol: NNDR for synthetic to holdout.
            - nndr_trn_hol: NNDR for training to holdout.
    """
    if hol_embeds is not None:
        assert trn_embeds.shape == hol_embeds.shape

    # calculate DCR / NNDR for synthetic to training
    dcr_syn_trn, nndr_syn_trn = calculate_dcrs_nndrs(data=trn_embeds, query=syn_embeds)
    # calculate DCR / NNDR for synthetic to holdout
    dcr_syn_hol, nndr_syn_hol = calculate_dcrs_nndrs(data=hol_embeds, query=syn_embeds)
    # calculate DCR / NNDR for holdout to training
    dcr_trn_hol, nndr_trn_hol = calculate_dcrs_nndrs(data=trn_embeds, query=hol_embeds)

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
