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

import numpy as np
from joblib import cpu_count

from mostlyai.qa.common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
)
from mostlyai.qa.filesystem import TemporaryWorkspace
from plotly import graph_objs as go
from sklearn.neighbors import NearestNeighbors

_LOG = logging.getLogger(__name__)


def calculate_distances(
    *, syn_embeds: np.ndarray, trn_embeds: np.ndarray, hol_embeds: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Calculates distances to the closest records (DCR). Once for synthetic to training, and once for synthetic to
    holdout data.
    """

    if hol_embeds is not None:
        assert trn_embeds.shape == hol_embeds.shape
    # calculate DCR using L2 metric
    index = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="l2", n_jobs=min(cpu_count() - 1, 16))
    index.fit(syn_embeds)
    _LOG.info(f"calculate DCRs for {len(syn_embeds):,} synthetic to {len(trn_embeds):,} training")
    dcrs_trn, _ = index.kneighbors(trn_embeds)
    dcr_trn = dcrs_trn[:, 0]
    if hol_embeds is not None:
        _LOG.info(f"calculate DCRs for {len(syn_embeds):,} synthetic to {len(hol_embeds):,} holdout")
        dcrs_hol, _ = index.kneighbors(hol_embeds)
        dcr_hol = dcrs_hol[:, 0]
    else:
        dcr_hol = None
    dcr_trn_deciles = np.round(np.quantile(dcr_trn, np.linspace(0, 1, 11)), 3)
    _LOG.info(f"DCR deciles for synthetic to training: {dcr_trn_deciles}")
    if dcr_hol is not None:
        dcr_hol_deciles = np.round(np.quantile(dcr_hol, np.linspace(0, 1, 11)), 3)
        _LOG.info(f"DCR deciles for synthetic to holdout:  {dcr_hol_deciles}")
        # calculate share of dcr_trn != dcr_hol
        _LOG.info(f"share of dcr_trn < dcr_hol: {np.mean(dcr_trn < dcr_hol):.1%}")
        _LOG.info(f"share of dcr_trn > dcr_hol: {np.mean(dcr_trn > dcr_hol):.1%}")
    return dcr_trn, dcr_hol


def plot_distances(plot_title: str, dcr_trn: np.ndarray, dcr_hol: np.ndarray | None) -> go.Figure:
    # calculate quantiles
    y = np.linspace(0, 1, 101)
    x_trn = np.quantile(dcr_trn, y)
    if dcr_hol is not None:
        x_hol = np.quantile(dcr_hol, y)
    else:
        x_hol = None
    # prepare layout
    layout = go.Layout(
        title=dict(text=f"<b>{plot_title}</b>", x=0.5, y=0.98),
        title_font=CHARTS_FONTS["title"],
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=500,
        margin=dict(l=20, r=20, b=20, t=40, pad=5),
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(
            showticklabels=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#999999",
            rangemode="tozero",
        ),
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=1,
        cols=1,
    )
    # plot content
    cum_trn_scatter = go.Scatter(
        mode="lines",
        x=x_trn,
        y=y,
        name="DCR training",
        line=dict(color=CHARTS_COLORS["synthetic"], width=5),
        yhoverformat=".0%",
    )
    fig.add_trace(cum_trn_scatter, row=1, col=1)
    if x_hol is not None:
        cum_hol_scatter = go.Scatter(
            mode="lines",
            x=x_hol,
            y=y,
            name="DCR holdout",
            line=dict(color=CHARTS_COLORS["original"], width=5),
            yhoverformat=".0%",
        )
        fig.add_trace(cum_hol_scatter, row=1, col=1)
    return fig


def plot_store_distances(
    dcr_trn: np.ndarray,
    dcr_hol: np.ndarray | None,
    workspace: TemporaryWorkspace,
) -> None:
    fig = plot_distances("Cumulative Distributions of Distance to Closest Records (DCR)", dcr_trn, dcr_hol)
    workspace.store_figure_html(fig, "distances_dcr")
