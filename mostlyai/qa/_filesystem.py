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

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import pandas as pd
from plotly import graph_objs as go

_OLD_COL_PREFIX = r"^(tgt|ctx|nxt)(\.|⁝)"
_NEW_COL_PREFIX = r"\1::"


class TemporaryWorkspace(TemporaryDirectory):
    FIGURE_TYPE = Literal[
        "univariate",
        "bivariate",
        "distinct_categories_per_sequence",
        "sequences_per_distinct_category",
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
        if figure_type in [
            "univariate",
            "bivariate",
            "distinct_categories_per_sequence",
            "sequences_per_distinct_category",
        ]:
            file = self.get_figure_path(figure_type, *cols)
        else:
            file = self.get_unique_figure_path(figure_type)
        self._store_figure_html(fig, file)


class Statistics:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.early_exit_path = self.path / "_EARLY_EXIT"
        self.meta_path = self.path / "meta.json"

        # correlation
        self.correlations_path = self.path / "correlations.parquet"

        # accuracy
        self.bins_dir = self.path / "bins"
        self.univariate_accuracies_path = self.path / "univariate_accuracies.parquet"
        self.bivariate_accuracies_path = self.path / "bivariate_accuracies.parquet"
        self.trivariate_accuracies_path = self.path / "trivariate_accuracies.parquet"
        self.numeric_kdes_uni_dir = self.path / "numeric_kdes_uni"
        self.categorical_counts_uni_dir = self.path / "categorical_counts_uni"
        self.bin_counts_uni_path = self.path / "bin_counts_uni.parquet"
        self.bin_counts_biv_path = self.path / "bin_counts_biv.parquet"

        # coherence
        self.coherence_bins_dir = self.path / "coherence_bins"
        self.distinct_categories_per_sequence_bins_dir = self.path / "distinct_categories_per_sequence_bins"
        self.distinct_categories_per_sequence_kdes_dir = self.path / "distinct_categories_per_sequence_kdes"
        self.distinct_categories_per_sequence_counts_dir = self.path / "distinct_categories_per_sequence_counts"
        self.distinct_categories_per_sequence_accuracy_path = (
            self.path / "distinct_categories_per_sequence_accuracy.parquet"
        )
        self.sequences_per_distinct_category_counts_dir = self.path / "sequences_per_distinct_category_counts"
        self.sequences_per_distinct_category_top_category_counts_dir = (
            self.path / "sequences_per_distinct_category_top_category_counts"
        )
        self.sequences_per_distinct_category_top_cats_dir = self.path / "sequences_per_distinct_category_top_cats"
        self.sequences_per_distinct_category_n_seqs_path = self.path / "sequences_per_distinct_category_n_seqs.parquet"
        self.sequences_per_distinct_category_accuracy_path = (
            self.path / "sequences_per_distinct_category_accuracy.parquet"
        )

        # similarity
        self.ori_pca_path = self.path / "trn_pca.npy"
        self.hol_pca_path = self.path / "hol_pca.npy"

    def _store_file_per_row(self, df: pd.DataFrame, path: Path, explode_cols: list[str]) -> None:
        path.mkdir(exist_ok=True, parents=True)
        for i, row in df.iterrows():
            row_df = pd.DataFrame([row]).explode(explode_cols)
            row_df.to_parquet(path / f"{i:05}.parquet")

    def _load_df_from_row_files(self, path: Path, cols: list[str], groupby_col: str) -> pd.DataFrame:
        files = sorted(path.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in files]) if files else pd.DataFrame(columns=cols)
        df = df.groupby(groupby_col, sort=False).agg(list).reset_index()
        return df

    def mark_early_exit(self) -> None:
        self.early_exit_path.touch()

    def is_early_exit(self) -> bool:
        return self.early_exit_path.exists()

    def store_meta(self, meta: dict):
        with open(self.meta_path, "w", encoding="utf-8") as file:
            json.dump(meta, file)

    def load_meta(self) -> dict:
        with open(self.meta_path, encoding="utf-8") as file:
            return json.load(file)

    def store_bins(self, bins: dict[str, list]) -> None:
        df = pd.Series(bins).to_frame("bins").reset_index().rename(columns={"index": "column"})
        self._store_file_per_row(df, self.bins_dir, ["bins"])

    def load_bins(self) -> dict[str, list]:
        df = self._load_df_from_row_files(self.bins_dir, ["column", "bins"], "column")
        # harmonise older prefix formats to <prefix>:: for compatibility with older versions
        df["column"] = df["column"].str.replace(_OLD_COL_PREFIX, _NEW_COL_PREFIX, regex=True)
        return df.set_index("column")["bins"].to_dict()

    def store_correlations(self, corr: pd.DataFrame) -> None:
        corr.to_parquet(self.correlations_path)

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

    def store_trivariate_accuracies(self, trivariates: pd.DataFrame) -> None:
        trivariates.to_parquet(self.trivariate_accuracies_path)

    def load_trivariate_accuracies(self) -> pd.DataFrame:
        if not self.trivariate_accuracies_path.exists():
            return pd.DataFrame(columns=["col1", "col2", "col3", "accuracy", "accuracy_max"])
        df = pd.read_parquet(self.trivariate_accuracies_path)
        return df

    def store_numeric_uni_kdes(self, trn_kdes: dict[str, pd.Series]) -> None:
        trn_kdes = pd.DataFrame(
            [(column, list(xy.index), list(xy.values)) for column, xy in trn_kdes.items()],
            columns=["column", "x", "y"],
        )
        self._store_file_per_row(trn_kdes, self.numeric_kdes_uni_dir, ["x", "y"])

    def load_numeric_uni_kdes(self) -> dict[str, pd.Series]:
        trn_kdes = self._load_df_from_row_files(self.numeric_kdes_uni_dir, ["column", "x", "y"], "column")
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
        self._store_file_per_row(trn_cnts_uni, self.categorical_counts_uni_dir, ["cat", "count"])

    def load_categorical_uni_counts(self) -> dict[str, pd.Series]:
        trn_cnts_uni = self._load_df_from_row_files(
            self.categorical_counts_uni_dir, ["column", "cat", "count"], "column"
        )
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
        ori_cnts_uni: dict[str, pd.Series],
        ori_cnts_biv: dict[tuple[str, str], pd.Series],
    ) -> None:
        # store univariate bin counts
        ori_cnts_uni = pd.DataFrame(
            [(column, list(bin_counts.index), list(bin_counts.values)) for column, bin_counts in ori_cnts_uni.items()],
            columns=["column", "bin", "count"],
        )
        ori_cnts_uni.to_parquet(self.bin_counts_uni_path)

        # store bivariate bin counts
        ori_cnts_biv = pd.DataFrame(
            [
                (column[0], column[1], list(bin_counts.index), list(bin_counts.values))
                for column, bin_counts in ori_cnts_biv.items()
            ],
            columns=["col1", "col2", "bin", "count"],
        )
        ori_cnts_biv.to_parquet(self.bin_counts_biv_path)

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

    def store_coherence_bins(self, bins: dict[str, list]) -> None:
        df = pd.Series(bins).to_frame("bins").reset_index().rename(columns={"index": "column"})
        self._store_file_per_row(df, self.coherence_bins_dir, ["bins"])

    def load_coherence_bins(self) -> dict[str, list] | None:
        if not self.coherence_bins_dir.exists():
            return None
        df = self._load_df_from_row_files(self.coherence_bins_dir, ["column", "bins"], "column")
        return df.set_index("column")["bins"].to_dict()

    def store_distinct_categories_per_sequence_kdes(self, ori_kdes: dict[str, pd.Series]) -> None:
        ori_kdes = pd.DataFrame(
            [(column, list(xy.index), list(xy.values)) for column, xy in ori_kdes.items()],
            columns=["column", "x", "y"],
        )
        self._store_file_per_row(ori_kdes, self.distinct_categories_per_sequence_kdes_dir, ["x", "y"])

    def load_distinct_categories_per_sequence_kdes(self) -> dict[str, pd.Series]:
        kdes = self._load_df_from_row_files(
            self.distinct_categories_per_sequence_kdes_dir, ["column", "x", "y"], "column"
        )
        kdes = {
            row["column"]: pd.Series(
                row["y"],
                index=row["x"],
                name=row["column"],
            )
            for _, row in kdes.iterrows()
        }
        return kdes

    def store_distinct_categories_per_sequence_bins(self, bins: dict[str, list]) -> None:
        df = pd.Series(bins).to_frame("bins").reset_index().rename(columns={"index": "column"})
        self._store_file_per_row(df, self.distinct_categories_per_sequence_bins_dir, ["bins"])

    def load_distinct_categories_per_sequence_bins(self) -> dict[str, list]:
        df = self._load_df_from_row_files(self.distinct_categories_per_sequence_bins_dir, ["column", "bins"], "column")
        return df.set_index("column")["bins"].to_dict()

    def store_binned_distinct_categories_per_sequence_counts(self, counts: dict[str, pd.Series]) -> None:
        counts = pd.DataFrame(
            [(column, list(counts.index), list(counts.values)) for column, counts in counts.items()],
            columns=["column", "cat", "count"],
        )
        self._store_file_per_row(counts, self.distinct_categories_per_sequence_counts_dir, ["cat", "count"])

    def load_binned_distinct_categories_per_sequence_counts(self) -> dict[str, pd.Series]:
        counts = self._load_df_from_row_files(
            self.distinct_categories_per_sequence_counts_dir, ["column", "cat", "count"], "column"
        )
        counts = {
            row["column"]: pd.Series(
                row["count"],
                index=row["cat"],
                name=row["column"],
            )
            for _, row in counts.iterrows()
        }
        return counts

    def store_distinct_categories_per_sequence_accuracy(self, accuracy: pd.DataFrame) -> None:
        accuracy.to_parquet(self.distinct_categories_per_sequence_accuracy_path)

    def load_distinct_categories_per_sequence_accuracy(self) -> pd.DataFrame:
        df = pd.read_parquet(self.distinct_categories_per_sequence_accuracy_path)
        return df

    def store_sequences_per_distinct_category_artifacts(
        self,
        seqs_per_cat_cnts: dict[str, pd.Series],
        seqs_per_top_cat_cnts: dict[str, pd.Series],
        top_cats: dict[str, list[str]],
        n_seqs: int,
    ) -> None:
        # store seqs_per_cat_cnts
        seqs_per_cat_cnts_df = pd.DataFrame(
            [
                (column, list(cat_counts.index), list(cat_counts.values))
                for column, cat_counts in seqs_per_cat_cnts.items()
            ],
            columns=["column", "cat", "count"],
        )
        self._store_file_per_row(
            seqs_per_cat_cnts_df, self.sequences_per_distinct_category_counts_dir, ["cat", "count"]
        )

        # store seqs_per_top_cat_cnts
        seqs_per_top_cat_cnts_df = pd.DataFrame(
            [
                (column, list(cat_counts.index), list(cat_counts.values))
                for column, cat_counts in seqs_per_top_cat_cnts.items()
            ],
            columns=["column", "cat", "count"],
        )
        self._store_file_per_row(
            seqs_per_top_cat_cnts_df, self.sequences_per_distinct_category_top_category_counts_dir, ["cat", "count"]
        )

        # store top_cats
        top_cats_df = pd.Series(top_cats).to_frame("top_cats").reset_index().rename(columns={"index": "column"})
        self._store_file_per_row(top_cats_df, self.sequences_per_distinct_category_top_cats_dir, ["top_cats"])

        # store n_seqs
        pd.Series({"n_seqs": n_seqs}).to_frame("n_seqs").to_parquet(self.sequences_per_distinct_category_n_seqs_path)

    def load_sequences_per_distinct_category_artifacts(
        self,
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, list[str]], int]:
        # load seqs_per_cat_cnts
        seqs_per_cat_cnts = self._load_df_from_row_files(
            self.sequences_per_distinct_category_counts_dir, ["column", "cat", "count"], "column"
        )
        seqs_per_cat_cnts = {
            row["column"]: pd.Series(
                row["count"],
                index=row["cat"],
                name=row["column"],
            )
            for _, row in seqs_per_cat_cnts.iterrows()
        }

        # load seqs_per_top_cat_cnts
        seqs_per_top_cat_cnts = self._load_df_from_row_files(
            self.sequences_per_distinct_category_top_category_counts_dir, ["column", "cat", "count"], "column"
        )
        seqs_per_top_cat_cnts = {
            row["column"]: pd.Series(
                row["count"],
                index=row["cat"],
                name=row["column"],
            )
            for _, row in seqs_per_top_cat_cnts.iterrows()
        }

        # load top_cats
        top_cats = self._load_df_from_row_files(
            self.sequences_per_distinct_category_top_cats_dir, ["column", "top_cats"], "column"
        )
        top_cats = top_cats.set_index("column")["top_cats"].to_dict()

        # load n_seqs
        n_seqs = pd.read_parquet(self.sequences_per_distinct_category_n_seqs_path)
        n_seqs = n_seqs["n_seqs"].iloc[0]

        return seqs_per_cat_cnts, seqs_per_top_cat_cnts, top_cats, n_seqs

    def store_sequences_per_distinct_category_accuracy(self, accuracy: pd.DataFrame) -> None:
        accuracy.to_parquet(self.sequences_per_distinct_category_accuracy_path)

    def load_sequences_per_distinct_category_accuracy(self) -> pd.DataFrame:
        df = pd.read_parquet(self.sequences_per_distinct_category_accuracy_path)
        return df
