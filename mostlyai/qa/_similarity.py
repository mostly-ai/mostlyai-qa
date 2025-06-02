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
import plotly.graph_objects as go
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from mostlyai.qa._common import (
    CHARTS_COLORS,
    CHARTS_FONTS,
)
from mostlyai.qa._filesystem import TemporaryWorkspace

_LOG = logging.getLogger(__name__)


def calculate_cosine_similarities(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
) -> tuple[np.float64 | None, np.float64]:
    # calculate centroids
    syn_centroid = syn_embeds.mean(axis=0, keepdims=True)
    trn_centroid = trn_embeds.mean(axis=0, keepdims=True)
    if hol_embeds is not None:
        hol_centroid = hol_embeds.mean(axis=0, keepdims=True)
    else:
        hol_centroid = None
    # calculate centroid similarities
    if hol_centroid is not None:
        sim_cosine_trn_hol = np.clip(cosine_similarity(trn_centroid, hol_centroid)[0][0], 0.0, 1.0)
        _LOG.info(f"calculated cosine similarity for trn and hol: {sim_cosine_trn_hol:.7f}")
    else:
        sim_cosine_trn_hol = None
    sim_cosine_trn_syn = np.clip(cosine_similarity(trn_centroid, syn_centroid)[0][0], 0.0, 1.0)
    _LOG.info(f"calculated cosine similarity for trn and syn: {sim_cosine_trn_syn:.7f}")
    return sim_cosine_trn_hol, sim_cosine_trn_syn


def calculate_discriminator_auc(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None,
) -> tuple[np.float64 | None, np.float64]:
    def calculate_mean_auc(embeds1, embeds2):
        """
        Calculate the mean AUC score using 10-fold cross-validation with a 90/10 split
        for a ML model to discriminate between two embedding arrays.
        """

        # limit the number of samples to 10000
        embeds1 = embeds1[:10000]
        embeds2 = embeds2[:10000]

        # create labels for the data
        labels1 = np.zeros(embeds1.shape[0])
        labels2 = np.ones(embeds2.shape[0])

        # combine the data and labels
        X = np.vstack((embeds1, embeds2))
        y = np.hstack((labels1, labels2))

        # initialize the cross-validator
        kf = StratifiedKFold(n_splits=10, shuffle=True)

        # initialize a list to store AUC scores
        auc_scores = []

        try:
            # perform 10-fold cross-validation
            for train_index, test_index in kf.split(X, y):
                X_train, X_holdout = X[train_index], X[test_index]
                y_train, y_holdout = y[train_index], y[test_index]

                # train a ML classifier
                clf = HistGradientBoostingClassifier(
                    max_iter=50,
                    max_depth=10,
                    min_samples_leaf=5,
                    max_features=0.5,
                )
                clf.fit(X_train, y_train)

                # predict probabilities on the holdout set
                y_holdout_pred = clf.predict_proba(X_holdout)[:, 1]

                # calculate the AUC score
                auc_score = roc_auc_score(y_holdout, y_holdout_pred)
                auc_scores.append(round(auc_score, 4))

            _LOG.info(f"auc_scores={[float(auc) for auc in auc_scores]}")

            # calculate the mean AUC score
            mean_auc_score = np.mean(auc_scores)

        except Exception as e:
            _LOG.warning(f"calculate_mean_auc failed: {e}")
            mean_auc_score = None

        return mean_auc_score

    if hol_embeds is not None:
        t0 = time.time()
        sim_auc_trn_hol = calculate_mean_auc(trn_embeds, hol_embeds)
        _LOG.info(f"calculated AUC for trn and hol: {sim_auc_trn_hol:.1%} in {time.time() - t0:.2f} seconds")
    else:
        sim_auc_trn_hol = None
    t0 = time.time()
    sim_auc_trn_syn = calculate_mean_auc(trn_embeds, syn_embeds)
    _LOG.info(f"calculated AUC for trn and syn: {sim_auc_trn_syn:.1%} in {time.time() - t0:.2f} seconds")
    return sim_auc_trn_hol, sim_auc_trn_syn


def make_contour_and_centroid_traces(
    data: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, color_scale: str
) -> tuple[go.Contour, go.Scatter]:
    # calculate the centroid
    centroid = data.mean(axis=0)

    # make grid over PCA space
    x = np.linspace(x_min - 0.05, x_max + 0.05, 100)
    y = np.linspace(y_min - 0.05, y_max + 0.05, 100)
    X, Y = np.meshgrid(x, y)

    # estimate gaussian kernels
    data = data.T
    # avoid singular matrix error by adding some noise
    noise = np.abs(minimum * 1e-3 if (minimum := np.min(data)) != 0 else 1e-18)
    data += np.random.normal(loc=0, scale=noise, size=data.shape)
    try:
        Z = scipy.stats.gaussian_kde(data)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except Exception as e:
        _LOG.warning(f"gaussian_kde failed, using ones instead: {e}")
        Z = np.ones_like(X)

    # make contour
    contour = go.Contour(
        z=Z,
        x=x,
        y=y,
        colorscale=color_scale,
        line=dict(width=0, color="rgba(0,0,0,0.0)"),
        contours=dict(
            coloring="fill",
            showlabels=False,
            start=0,
            end=np.max(Z),
            size=np.max(Z) / 10,
            labelfont=dict(size=12, color="white"),
        ),
        opacity=0.8,
        showscale=False,
        hoverinfo="skip",
    )

    # make centroid marker
    centroid_marker = go.Scatter(
        x=[centroid[0]],
        y=[centroid[1]],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Centroid",
        hoverinfo="skip",
        opacity=1.0,
    )

    return contour, centroid_marker


def plot_store_similarity_contours(
    *,
    syn_embeds: np.ndarray,
    trn_embeds: np.ndarray,
    hol_embeds: np.ndarray | None = None,
    workspace: TemporaryWorkspace = None,
):
    if trn_embeds.shape[1] < 3:
        return

    # limit the number of samples to 10000
    syn_embeds = syn_embeds[:10000]
    trn_embeds = trn_embeds[:10000]
    hol_embeds = hol_embeds[:10000] if hol_embeds is not None else None

    # perform PCA on trn embeddings
    pca_model = PCA(n_components=3)
    pca_model.fit(trn_embeds)

    # transform embeddings to PCA space
    syn_pca = pca_model.transform(syn_embeds)
    trn_pca = pca_model.transform(trn_embeds)
    hol_pca = pca_model.transform(hol_embeds) if hol_embeds is not None else None

    # calculate percentiles to make axis ranges resilient towards outliers
    pcas = [trn_pca, syn_pca] + ([hol_pca] if hol_pca is not None else [])
    pca_min = np.quantile(np.vstack(pcas), 0.02, axis=0)
    pca_max = np.quantile(np.vstack(pcas), 0.98, axis=0)

    # make plots layout
    pca_combos = [(1, 0), (2, 0)]
    n_rows = len(pca_combos)
    layout = go.Layout(
        font=CHARTS_FONTS["base"],
        hoverlabel=CHARTS_FONTS["hover"],
        plot_bgcolor=CHARTS_COLORS["background"],
        autosize=True,
        height=400 * n_rows,
        margin=dict(l=10, r=10, b=10, t=50, pad=5),
        showlegend=False,
        dragmode=False,
    )
    fig = go.Figure(layout=layout).set_subplots(
        rows=n_rows,
        cols=len(pcas),
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=["training", "synthetic", "holdout"][: len(pcas)],
    )
    fig.update_annotations(font_size=12)
    fig.update_xaxes(
        showgrid=True, gridcolor="black", gridwidth=0.5, zeroline=True, zerolinecolor="black", zerolinewidth=2.0
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="black", gridwidth=0.5, zeroline=True, zerolinecolor="black", zerolinewidth=2.0
    )

    color_scales = ["Blues", "Greens", "Blues"][: len(pcas)]
    for row, (pca_y, pca_x) in enumerate(pca_combos):
        # update axes labels of subplots
        label_x, label_y = f"Principal Component {pca_x + 1}", f"Principal Component {pca_y + 1}"
        fig.update_xaxes(row=row + 1, title_text=label_x, showgrid=True, zeroline=True)
        # add contours and centroid traces to subplots
        for col, pca in enumerate(pcas):
            if col == 0:
                fig.update_yaxes(row=row + 1, col=col + 1, title_text=label_y, showgrid=True, zeroline=True)
            else:
                fig.update_yaxes(row=row + 1, col=col + 1, title_text="", showgrid=True, zeroline=True)
            contour, centroid = make_contour_and_centroid_traces(
                data=pca[:, [pca_x, pca_y]],
                x_min=pca_min[pca_x],
                x_max=pca_max[pca_x],
                y_min=pca_min[pca_y],
                y_max=pca_max[pca_y],
                color_scale=color_scales[col],
            )
            fig.add_trace(contour, row=row + 1, col=col + 1)
            fig.add_trace(centroid, row=row + 1, col=col + 1)

    # store the figure
    workspace.store_figure_html(fig, "similarity_pca")
