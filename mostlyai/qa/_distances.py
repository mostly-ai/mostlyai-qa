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

import numpy as np
from joblib import cpu_count

from mostlyai.qa._common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
)
from mostlyai.qa._filesystem import TemporaryWorkspace
from plotly import graph_objs as go
from sklearn.neighbors import NearestNeighbors

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
    # sort data by first dimension to enforce deterministic results
    data = data[data[:, 0].argsort()]
    _LOG.info(f"calculate DCRs for {data.shape=} and {query.shape=}")
    index = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="cosine", n_jobs=min(cpu_count() - 1, 16))
    index.fit(data)
    dcrs, _ = index.kneighbors(query)
    dcr = dcrs[:, 0]
    nndr = (dcrs[:, 0] + 1e-8) / (dcrs[:, 1] + 1e-8)
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

    # calculate quantiles
    y = np.linspace(0, 1, 101)
    x_syn_trn = np.quantile(dcr_syn_trn, y)
    if dcr_syn_hol is not None:
        x_syn_hol = np.quantile(dcr_syn_hol, y)
    else:
        x_syn_hol = None

    if dcr_trn_hol is not None:
        x_trn_hol = np.quantile(dcr_trn_hol, y)
    else:
        x_trn_hol = None

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
        margin=dict(l=20, r=20, b=20, t=40, pad=5),
        showlegend=True,
        yaxis=dict(
            showticklabels=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#999999",
            rangemode="tozero",
            showline=True,
            linewidth=1,
            linecolor="#999999",
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            tickformat=".0%",
            showgrid=False,
            range=[0, 1],
            showline=True,
            linewidth=1,
            linecolor="#999999",
        ),
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="#999999",
            hoverformat=".3f",
        ),
    )
    fig = go.Figure(layout=layout)

    traces = []

    # training vs holdout (light gray)
    if x_trn_hol is not None:
        traces.append(
            go.Scatter(
                mode="lines",
                x=x_trn_hol,
                y=y,
                name="Training vs. Holdout Data",
                line=dict(color="#999999", width=5),
                yaxis="y2",
            )
        )

    # synthetic vs holdout (gray)
    if x_syn_hol is not None:
        traces.append(
            go.Scatter(
                mode="lines",
                x=x_syn_hol,
                y=y,
                name="Synthetic vs. Holdout Data",
                line=dict(color="#666666", width=5),
                yaxis="y2",
            )
        )

    # synthetic vs training (green)
    traces.append(
        go.Scatter(
            mode="lines",
            x=x_syn_trn,
            y=y,
            name="Synthetic vs. Training Data",
            line=dict(color="#24db96", width=5),
            yaxis="y2",
        )
    )

    for trace in traces:
        fig.add_trace(trace)

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
        "Cumulative Distributions of Distance to Closest Records (DCR)",
        distances,
    )
    workspace.store_figure_html(fig, "distances_dcr")
