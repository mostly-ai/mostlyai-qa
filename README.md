# Synthetic Data - Quality Assurance

[![](https://pepy.tech/badge/mostlyai-qa)](https://pypi.org/project/mostlyai-qa/) ![](https://img.shields.io/github/license/mostly-ai/mostlyai-qa) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-qa) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-qa)

Assess the fidelity and novelty of synthetic samples with respect to original samples:

1. calculate a rich set of accuracy, similarity and distance metrics
2. visualize statistics for easy comparison to training and holdout samples
3. generate a standalone, easy-to-share, easy-to-read HTML summary report

...all with a single line of Python code 💥.

## Installation

The latest release of `mostlyai-qa` can be installed via pip:

```bash
pip install -U mostlyai-qa
```

## Quick start

```python
import pandas as pd
import webbrowser
import json
from mostlyai import qa

# fetch original + synthetic data
base_url = 'https://github.com/mostly-ai/mostlyai-qa/raw/refs/heads/main/examples/quick-start'
syn = pd.read_csv(f'{base_url}/census2k-syn_mostly.csv.gz')
# syn = pd.read_csv(f'{base_url}/census2k-syn_flip30.csv.gz') # a 30% perturbation of trn
trn = pd.read_csv(f'{base_url}/census2k-trn.csv.gz')
hol = pd.read_csv(f'{base_url}/census2k-hol.csv.gz')

# runs for ~30secs
report_path, metrics = qa.report(
    syn_tgt_data = syn,
    trn_tgt_data = trn,
    hol_tgt_data = hol,
)

# pretty print metrics
print(json.dumps(metrics, indent=4))

# open up HTML report in new browser window
webbrowser.open(f"file://{report_path.absolute()}")
```

## Basic usage

```python
from mostlyai import qa

# analyze single-table data
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
)

# analyze sequential data
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
    tgt_context_key = "user_id",
)

# analyze sequential data with context
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
    syn_ctx_data = synthetic_context_df,
    trn_ctx_data = training_context_df,
    hol_ctx_data = holdout_context_df,  # optional
    ctx_primary_key = "id",
    tgt_context_key = "user_id",
)
```

Note, that due to the calculation of embeddings the function call might take a while. Embedding 10k samples on a Mac M2 take for example about 40secs. Limit the size of the passed DataFrames, or use the `max_sample_size_embeddings` parameter to speed up the report.

## Function signature

```python
def report(
    *,
    syn_tgt_data: pd.DataFrame,
    trn_tgt_data: pd.DataFrame,
    hol_tgt_data: pd.DataFrame | None = None,
    syn_ctx_data: pd.DataFrame | None = None,
    trn_ctx_data: pd.DataFrame | None = None,
    hol_ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    report_path: str | Path | None = "model-report.html",
    report_title: str = "Model Report",
    report_subtitle: str = "",
    report_credits: str = REPORT_CREDITS,
    report_extra_info: str = "",
    max_sample_size_accuracy: int | None = None,
    max_sample_size_embeddings: int | None = None,
    statistics_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, dict | None]:
    """
    Generate HTML report and metrics for comparing synthetic and original data samples.

    Args:
        syn_tgt_data: Synthetic samples
        trn_tgt_data: Training samples
        hol_tgt_data: Holdout samples
        syn_ctx_data: Synthetic context samples
        trn_ctx_data: Training context samples
        hol_ctx_data: Holdout context samples
        ctx_primary_key: Column within the context data that contains the primary key
        tgt_context_key: Column within the target data that contains the key to link to the context
        report_path: Path of where to store the HTML report
        report_title: Title of the HTML report
        report_subtitle: Subtitle of the HTML report
        report_credits: Credits of the HTML report
        report_extra_info: Extra details to be included to the HTML report
        max_sample_size_accuracy: Max sample size for accuracy
        max_sample_size_embeddings: Max sample size for embeddings (similarity & distances)
        statistics_path: Path of where to store the statistics to be used by `report_from_statistics`
        on_progress: A custom progress callback
    Returns:
        1. Path to the HTML report
        2. Dictionary of calculated metrics:
        - `accuracy`:  # Accuracy is defined as (100% - Total Variation Distance), for each distribution, and then averaged across.
          - `overall`: Overall accuracy of synthetic data, i.e. average across univariate, bivariate and coherence.
          - `univariate`: Average accuracy of discretized univariate distributions.
          - `bivariate`: Average accuracy of discretized bivariate distributions.
          - `coherence`: Average accuracy of discretized coherence distributions. Only applicable for sequential data.
          - `overall_max`: Expected overall accuracy of a same-sized holdout. Serves as reference for `overall`.
          - `univariate_max`: Expected univariate accuracy of a same-sized holdout. Serves as reference for `univariate`.
          - `bivariate_max`: Expected bivariate accuracy of a same-sized holdout. Serves as reference for `bivariate`.
          - `coherence_max`: Expected coherence accuracy of a same-sized holdout. Serves as reference for `coherence`.
        - `similarity`:  # All similarity metrics are calculated within an embedding space.
            - `cosine_similarity_training_synthetic`: Cosine similarity between training and synthetic centroids.
            - `cosine_similarity_training_holdout`: Cosine similarity between training and holdout centroids. Serves as reference for `cosine_similarity_training_synthetic`.
            - `discriminator_auc_training_synthetic`: Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples.
            - `discriminator_auc_training_holdout`: Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as reference for `discriminator_auc_training_synthetic`.
        - `distances`:  # All distance metrics are calculated within an embedding space. An equal number of training and holdout samples is considered.
            - `ims_training`: Share of synthetic samples that are identical to a training sample.
            - `ims_holdout`: Share of synthetic samples that are identical to a holdout sample. Serves as reference for `ims_training`.
            - `dcr_training`: Average L2 nearest-neighbor distance between synthetic and training samples.
            - `dcr_holdout`: Average L2 nearest-neighbor distance between synthetic and holdout samples. Serves as reference for `dcr_training`.
            - `dcr_share`: Share of synthetic samples that are closer to a training sample than to a holdout sample. This shall not be significantly larger than 50\%.
    """
```

## Metrics

Three sets of metrics are calculated to compare synthetic data with the original data.

### Accuracy

The L1 distances between the discretized marginal distributions of the synthetic and the original training data are being calculated for all columns.
The reported accuracy is expressed as 100% minus the total variational distance (TVD), which is half the L1 distance between the two distributions.
These accuracies are then averaged to produce a single accuracy score, where higher scores indicate better synthetic data.

1. **Univariate Accuracy**: The accuracy of the univariate distributions for all target columns is measured.
2. **Bivariate Accuracy**: The accuracy of all pair-wise distributions for target columns, as well as for target columns with respect to the context columns, is measured.
3. **Coherence Accuracy**: The accuracy of the auto-correlation for all target columns is measured. This is applicable only for sequential data.

An overall accuracy score is calculated as the average of these aggregate-level scores.

### Similarity

All records are embedded into an embedding space to calculate two metrics:

1. **Cosine Similarity**: The cosine similarity between the centroids of the synthetic and the original training data is calculated and compared to the cosine similarity between the centroids of the original training and holdout data. Higher scores indicate better synthetic data.
2. **Discriminator AUC**: A binary classifier is trained to determine whether synthetic and original training data can be distinguished based on their embeddings. This score is compared to the same metric for the original training and holdout data. A score close to 50% indicates that synthetic samples are indistinguishable from original samples.

### Distances

All records are embedded into an embedding space, and individual-level L2 distances between samples are measured. For each synthetic sample, the distance to the nearest original sample (DCR) is calculated. This is done once with respect to original training records and once with respect to holdout records. These DCRs are then compared. For privacy-safe synthetic data, it is expected that synthetic data is as close to original training data as it is to original holdout data.

## Sample HTML Report

![Metrics](./docs/screenshots/metrics.png)
![Accuracy Univariates](./docs/screenshots/accuracy_univariates.png)
![Accuracy Bivariates](./docs/screenshots/accuracy_bivariates.png)
![Accuracy Coherence](./docs/screenshots/accuracy_coherence.png)
![Similarity](./docs/screenshots/similarity.png)
![Distances](./docs/screenshots/distances.png)

See [here](./examples/) for further examples.

## Citation

Please consider citing our project if you find it useful:

```bibtex
@software{mostlyai-qa,
    author = {{MOSTLY AI}},
    title = {{MOSTLY AI - Quality Assurance}},
    url = {https://github.com/mostly-ai/mostlyai-qa},
    year = {2024}
}
```
