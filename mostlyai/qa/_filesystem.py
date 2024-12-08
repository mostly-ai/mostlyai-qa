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

import hashlib
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from sklearn.decomposition import PCA


_OLD_COL_PREFIX = r"^(tgt|ctx|nxt)(\.|⁝)"
_NEW_COL_PREFIX = r"\1::"


class TemporaryWorkspace(TemporaryDirectory):
    FIGURE_TYPE = Literal[
        "univariate",
        "bivariate",
        "accuracy_matrix",
        "correlation_matrices",
        "similarity_pca",
        "distances_dcr",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace_dir = Path(self.name)

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def get_figure_path(self, figure_type: FIGURE_TYPE, *cols: str) -> Path:
        # in order to prevent issues with filenames we use a hashed figure_id as a safe file name
        source = "__".join([figure_type] + list(cols)).encode()
        figure_id = hashlib.md5(source).hexdigest()
        return self.workspace_dir / "figures" / figure_type / f"{figure_id}.html"

    def get_figure_paths(self, figure_type: FIGURE_TYPE, cols_df: pd.DataFrame) -> dict:
        return {tuple(cols): self.get_figure_path(figure_type, *cols) for _, cols in cols_df.iterrows()}

    def get_unique_figure_path(self, figure_type: FIGURE_TYPE) -> Path:
        return self.workspace_dir / "figures" / f"{figure_type}.html"

    @staticmethod
    def _store_figure_html(fig: go.Figure, file: Path) -> None:
        file.parent.mkdir(exist_ok=True, parents=True)
        fig.write_html(
            file,
            full_html=False,
            include_plotlyjs=False,
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom",
                    "pan",
                    "select",
                    "zoomIn",
                    "zoomOut",
                    "autoScale",
                    "resetScale",
                ],
            },
        )

    def store_figure_html(self, fig: go.Figure, figure_type: FIGURE_TYPE, *cols: str) -> None:
        if figure_type in ["univariate", "bivariate"]:
            file = self.get_figure_path(figure_type, *cols)
        else:
            file = self.get_unique_figure_path(figure_type)
        self._store_figure_html(fig, file)


class Statistics:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.early_exit_path = self.path / "_EARLY_EXIT"
        self.meta_path = self.path / "meta.json"
        self.bins_path = self.path / "bins.pickle"
        self.correlations_path = self.path / "correlations.parquet"
        self.univariate_accuracies_path = self.path / "univariate_accuracies.parquet"
        self.bivariate_accuracies_path = self.path / "bivariate_accuracies.parquet"
        self.numeric_kdes_uni_path = self.path / "numeric_kdes_uni.pickle"
        self.categorical_counts_uni_path = self.path / "categorical_counts_uni.pickle"
        self.bin_counts_uni_path = self.path / "bin_counts_uni.parquet"
        self.bin_counts_biv_path = self.path / "bin_counts_biv.parquet"
        self.pca_model_path = self.path / "pca_model.pickle"
        self.trn_pca_path = self.path / "trn_pca.npy"
        self.hol_pca_path = self.path / "hol_pca.npy"

    def mark_early_exit(self) -> None:
        self.early_exit_path.touch()

    def is_early_exit(self) -> bool:
        return self.early_exit_path.exists()

    def store_meta(self, meta: dict):
        with open(self.meta_path, "w") as file:
            json.dump(meta, file)

    def load_meta(self) -> dict:
        with open(self.meta_path, "r") as file:
            return json.load(file)

    def store_bins(self, bins: dict[str, list]) -> None:
        df = pd.Series(bins).to_frame("bins").reset_index().rename(columns={"index": "column"})
        df.to_pickle(self.bins_path)

    def load_bins(self) -> dict[str, list]:
        df = pd.read_pickle(self.bins_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        df["column"] = df["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        return df.set_index("column")["bins"].to_dict()

    def store_correlations(self, trn_corr: pd.DataFrame) -> None:
        trn_corr.to_parquet(self.correlations_path)

    def load_correlations(self) -> pd.DataFrame:
        df = pd.read_parquet(self.correlations_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        df.index = df.index.str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        df.columns = df.columns.str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        return df

    def store_univariate_accuracies(self, univariates: pd.DataFrame) -> None:
        univariates.to_parquet(self.univariate_accuracies_path)

    def load_univariate_accuracies(self) -> pd.DataFrame:
        df = pd.read_parquet(self.univariate_accuracies_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        df["column"] = df["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        return df

    def store_bivariate_accuracies(self, bivariates: pd.DataFrame) -> None:
        bivariates.to_parquet(self.bivariate_accuracies_path)

    def load_bivariate_accuracies(self) -> pd.DataFrame:
        df = pd.read_parquet(self.bivariate_accuracies_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        df["col1"] = df["col1"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        df["col2"] = df["col2"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        return df

    def store_numeric_uni_kdes(self, trn_kdes: dict[str, pd.Series]) -> None:
        trn_kdes = pd.DataFrame(
            [(column, list(xy.index), list(xy.values)) for column, xy in trn_kdes.items()],
            columns=["column", "x", "y"],
        )
        trn_kdes.to_pickle(self.numeric_kdes_uni_path)

    def load_numeric_uni_kdes(self) -> dict[str, pd.Series]:
        trn_kdes = pd.read_pickle(self.numeric_kdes_uni_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        trn_kdes["column"] = trn_kdes["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        trn_kdes = {
            row["column"]: pd.Series(
                row["y"],
                index=row["x"],
                name=row["column"],
            )
            for _, row in trn_kdes.iterrows()
        }
        return trn_kdes

    def store_categorical_uni_counts(self, trn_cnts_uni: dict[str, pd.Series]) -> None:
        trn_cnts_uni = pd.DataFrame(
            [(column, list(cat_counts.index), list(cat_counts.values)) for column, cat_counts in trn_cnts_uni.items()],
            columns=["column", "cat", "count"],
        )
        trn_cnts_uni.to_pickle(self.categorical_counts_uni_path)

    def load_categorical_uni_counts(self) -> dict[str, pd.Series]:
        trn_cnts_uni = pd.read_pickle(self.categorical_counts_uni_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        trn_cnts_uni["column"] = trn_cnts_uni["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        trn_cnts_uni = {
            row["column"]: pd.Series(
                row["count"],
                index=row["cat"],
                name=row["column"],
            )
            for _, row in trn_cnts_uni.iterrows()
        }
        return trn_cnts_uni

    def store_bin_counts(
        self,
        trn_cnts_uni: dict[str, pd.Series],
        trn_cnts_biv: dict[tuple[str, str], pd.Series],
    ) -> None:
        # store univariate bin counts
        trn_cnts_uni = pd.DataFrame(
            [(column, list(bin_counts.index), list(bin_counts.values)) for column, bin_counts in trn_cnts_uni.items()],
            columns=["column", "bin", "count"],
        )
        trn_cnts_uni.to_parquet(self.bin_counts_uni_path)

        # store bivariate bin counts
        trn_cnts_biv = pd.DataFrame(
            [
                (column[0], column[1], list(bin_counts.index), list(bin_counts.values))
                for column, bin_counts in trn_cnts_biv.items()
            ],
            columns=["col1", "col2", "bin", "count"],
        )
        trn_cnts_biv.to_parquet(self.bin_counts_biv_path)

    def load_bin_counts(
        self,
    ) -> tuple[dict[str, pd.Series], dict[tuple[str, str], pd.Series]]:
        # load univariate bin counts
        trn_cnts_uni = pd.read_parquet(self.bin_counts_uni_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        trn_cnts_uni["column"] = trn_cnts_uni["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        trn_cnts_uni = {
            row["column"]: pd.Series(
                row["count"],
                index=pd.CategoricalIndex(row["bin"], categories=row["bin"], ordered=True),
                name=row["column"],
            )
            for _, row in trn_cnts_uni.iterrows()
        }

        # load bivariate bin counts
        def biv_multi_index(bin, col1, col2):
            bin = np.stack(bin)  # make it 2d numpy array
            col1_idx = pd.Series(bin[:, 0], name=col1, dtype="category").cat.reorder_categories(
                dict.fromkeys(bin[:, 0]), ordered=True
            )
            col2_idx = pd.Series(bin[:, 1], name=col2, dtype="category").cat.reorder_categories(
                dict.fromkeys(bin[:, 1]), ordered=True
            )
            return pd.MultiIndex.from_frame(pd.concat([col1_idx, col2_idx], axis=1))

        trn_cnts_biv = pd.read_parquet(self.bin_counts_biv_path)
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        trn_cnts_biv["col1"] = trn_cnts_biv["col1"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        trn_cnts_biv["col2"] = trn_cnts_biv["col2"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        trn_cnts_biv = {
            (row["col1"], row["col2"]): pd.Series(
                row["count"],
                index=biv_multi_index(row["bin"], row["col1"], row["col2"]),
            )
            for _, row in trn_cnts_biv.iterrows()
        }
        return trn_cnts_uni, trn_cnts_biv

    def store_pca_model(self, pca_model: PCA):
        with self.pca_model_path.open("wb") as file:
            pickle.dump(pca_model, file)

    def load_pca_model(self) -> PCA | None:
        if not self.pca_model_path.exists():
            return None
        with self.pca_model_path.open("rb") as file:
            return pickle.load(file)

    def store_trn_hol_pcas(self, trn_pca: np.ndarray, hol_pca: np.ndarray | None):
        np.save(self.trn_pca_path, trn_pca)
        if hol_pca is not None:
            np.save(self.hol_pca_path, hol_pca)

    def load_trn_hol_pcas(self) -> tuple[np.ndarray, np.ndarray | None]:
        trn_pca = np.load(self.trn_pca_path)
        if self.hol_pca_path.exists():
            hol_pca = np.load(self.hol_pca_path)
        else:
            hol_pca = None
        return trn_pca, hol_pca
