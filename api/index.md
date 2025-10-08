## QA Reference

### mostlyai.qa.report

```python
report(
    *,
    syn_tgt_data,
    trn_tgt_data,
    hol_tgt_data=None,
    syn_ctx_data=None,
    trn_ctx_data=None,
    hol_ctx_data=None,
    ctx_primary_key=None,
    tgt_context_key=None,
    report_path="model-report.html",
    report_title="Model Report",
    report_subtitle="",
    report_credits=REPORT_CREDITS,
    max_sample_size_accuracy=None,
    max_sample_size_coherence=None,
    max_sample_size_embeddings=None,
    statistics_path=None,
    update_progress=None
)

```

Generate an HTML report and metrics for assessing synthetic data quality.

Compares synthetic data samples with original training samples in terms of accuracy, similarity and distances. Provide holdout samples to calculate reference values for similarity and distances (recommended).

If synthetic data has been generated conditionally on a context dataset, provide the context data as well. This will allow for bivariate accuracy metrics between context and target to be calculated.

If the data represents sequential data, provide the `tgt_context_key` to set the groupby column for the target data.

Customize the report with the `report_title`, `report_subtitle` and `report_credits`.

Limit the compute time used by setting `max_sample_size_accuracy`, `max_sample_size_coherence` and `max_sample_size_embeddings`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `syn_tgt_data` | `DataFrame` | The synthetic (target) data. | *required* | | `trn_tgt_data` | `DataFrame` | The training (target) data. | *required* | | `hol_tgt_data` | `DataFrame | None` | The holdout (target) data. | `None` | | `syn_ctx_data` | `DataFrame | None` | The synthetic context data. | `None` | | `trn_ctx_data` | `DataFrame | None` | The training context data. | `None` | | `hol_ctx_data` | `DataFrame | None` | The holdout context data. | `None` | | `ctx_primary_key` | `str | None` | The primary key of the context data. | `None` | | `tgt_context_key` | `str | None` | The context key of the target data. | `None` | | `report_path` | `str | Path | None` | The path to store the HTML report. | `'model-report.html'` | | `report_title` | `str` | The title of the report. | `'Model Report'` | | `report_subtitle` | `str` | The subtitle of the report. | `''` | | `report_credits` | `str` | The credits of the report. | `REPORT_CREDITS` | | `max_sample_size_accuracy` | `int | None` | The maximum sample size for accuracy calculations. | `None` | | `max_sample_size_coherence` | `int | None` | The maximum sample size for coherence calculations. | `None` | | `max_sample_size_embeddings` | `int | None` | The maximum sample size for embedding calculations. | `None` | | `statistics_path` | `str | Path | None` | The path of where to store the statistics to be used by report_from_statistics | `None` | | `update_progress` | `ProgressCallback | None` | The progress callback. | `None` |

Returns:

| Type | Description | | --- | --- | | `Path` | The path to the generated HTML report. | | `ModelMetrics | None` | Metrics instance with accuracy, similarity, and distances metrics. |

### mostlyai.qa.report_from_statistics

```python
report_from_statistics(
    *,
    syn_tgt_data,
    syn_ctx_data=None,
    statistics_path=None,
    ctx_primary_key=None,
    tgt_context_key=None,
    report_path="data-report.html",
    report_title="Data Report",
    report_subtitle="",
    report_credits=REPORT_CREDITS,
    max_sample_size_accuracy=None,
    max_sample_size_coherence=None,
    update_progress=None
)

```

Generate an HTML report based on previously generated statistics and newly provided synthetic data samples.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `syn_tgt_data` | `DataFrame` | The synthetic (target) data. | *required* | | `syn_ctx_data` | `DataFrame | None` | The synthetic context data. | `None` | | `statistics_path` | `str | Path | None` | The path from where to fetch the statistics files. | `None` | | `ctx_primary_key` | `str | None` | The primary key of the context data. | `None` | | `tgt_context_key` | `str | None` | The context key of the target data. | `None` | | `report_path` | `str | Path | None` | The path to store the HTML report. | `'data-report.html'` | | `report_title` | `str` | The title of the report. | `'Data Report'` | | `report_subtitle` | `str` | The subtitle of the report. | `''` | | `report_credits` | `str` | The credits of the report. | `REPORT_CREDITS` | | `max_sample_size_accuracy` | `int | None` | The maximum sample size for accuracy calculations. | `None` | | `max_sample_size_coherence` | `int | None` | The maximum sample size for coherence calculations. | `None` | | `update_progress` | `ProgressCallback | None` | The progress callback. | `None` |

Returns:

| Type | Description | | --- | --- | | `Path` | The path to the generated HTML report. |

## Metrics Reference

### mostlyai.qa.metrics.ModelMetrics

Metrics regarding the quality of synthetic data, measured in terms of accuracy, similarity, and distances.

1. **Accuracy**: Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional marginal distributions.
1. **Similarity**: Metrics regarding the similarity of the full joint distributions of samples within an embedding space.
1. **Distances**: Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data.

The quality of synthetic data is assessed by comparing these metrics to the same metrics of a holdout dataset. The holdout dataset is a subset of the original training data, that was not used for training the synthetic data generator. The metrics of the synthetic data should be as close as possible to the metrics of the holdout data.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `accuracy` | `Accuracy | None` | Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional marginal distributions. | `None` | | `similarity` | `Similarity | None` | Metrics regarding the similarity of the full joint distributions of samples within an embedding space. | `None` | | `distances` | `Distances | None` | Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data. | `None` |

### mostlyai.qa.metrics.Accuracy

Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional marginal distributions.

1. **Univariate Accuracy**: The accuracy of the univariate distributions for all target columns.
1. **Bivariate Accuracy**: The accuracy of all pair-wise distributions for target columns, as well as for target columns with respect to the context columns.
1. **Trivariate Accuracy**: The accuracy of all three-way distributions for target columns.
1. **Coherence Accuracy**: The accuracy of the auto-correlation for all target columns.

Accuracy is defined as 100% - [Total Variation Distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) (TVD), whereas TVD is half the sum of the absolute differences of the relative frequencies of the corresponding distributions.

These accuracies are calculated for all discretized univariate, and bivariate distributions. In case of sequential data, also for all coherence distributions. Overall metrics are then calculated as the average across these accuracies.

All metrics can be compared against a theoretical maximum accuracy, which is calculated for a same-sized holdout. The accuracy metrics shall be as close as possible to the theoretical maximum, but not significantly higher, as this would indicate overfitting.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `overall` | `float | None` | Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence. | `None` | | `univariate` | `float | None` | Average accuracy of discretized univariate distributions. | `None` | | `bivariate` | `float | None` | Average accuracy of discretized bivariate distributions. | `None` | | `trivariate` | `float | None` | Average accuracy of discretized trivariate distributions. | `None` | | `coherence` | `float | None` | Average accuracy of discretized coherence distributions. Only applicable for sequential data. | `None` | | `overall_max` | `float | None` | Expected overall accuracy of a same-sized holdout. Serves as a reference for overall. | `None` | | `univariate_max` | `float | None` | Expected univariate accuracy of a same-sized holdout. Serves as a reference for univariate. | `None` | | `bivariate_max` | `float | None` | Expected bivariate accuracy of a same-sized holdout. Serves as a reference for bivariate. | `None` | | `trivariate_max` | `float | None` | Expected trivariate accuracy of a same-sized holdout. Serves as a reference for trivariate. | `None` | | `coherence_max` | `float | None` | Expected coherence accuracy of a same-sized holdout. Serves as a reference for coherence. | `None` |

### mostlyai.qa.metrics.Similarity

Metrics regarding the similarity of the full joint distributions of samples within an embedding space.

1. **Cosine Similarity**: The cosine similarity between the centroids of synthetic and training samples.
1. **Discriminator AUC**: The AUC of a discriminative model to distinguish between synthetic and training samples.

The Model2Vec model [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) is used to compute the embeddings of a string-ified representation of individual records. In case of sequential data the records, that belong to the same group, are being concatenated. We then calculate the cosine similarity between the centroids of the provided datasets within the embedding space.

Again, we expect the similarity metrics to be as close as possible to 1, but not significantly higher than what is measured for the holdout data, as this would again indicate overfitting.

In addition, a discriminative ML model is trained to distinguish between training and synthetic samples. The ability of this model to distinguish between training and synthetic samples is measured by the AUC score. For synthetic data to be considered realistic, the AUC score should be close to 0.5, which indicates that the synthetic data is indistinguishable from the training data.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `cosine_similarity_training_synthetic` | `float | None` | Cosine similarity between training and synthetic centroids. | `None` | | `cosine_similarity_training_holdout` | `float | None` | Cosine similarity between training and holdout centroids. Serves as a reference for cosine_similarity_training_synthetic. | `None` | | `discriminator_auc_training_synthetic` | `float | None` | Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples. | `None` | | `discriminator_auc_training_holdout` | `float | None` | Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as a reference for discriminator_auc_training_synthetic. | `None` |

### mostlyai.qa.metrics.Distances

Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numerically encoded space. Useful for assessing the novelty / privacy of synthetic data.

The provided data is first down-sampled, so that the number of samples match across all datasets. Note, that for an optimal sensitivity of this privacy assessment it is recommended to use a 50/50 split between training and holdout data, and then generate synthetic data of the same size.

The numerical encodings of these samples are then computed, and the nearest neighbor distances are calculated for each synthetic sample to the training and holdout samples. Based on these nearest neighbor distances the following metrics are calculated:

- Identical Match Share (IMS): The share of synthetic samples that are identical to a training or holdout sample.
- Distance to Closest Record (DCR): The average distance of synthetic to training or holdout samples.
- Nearest Neighbor Distance Ratio (NNDR): The 10-th smallest ratio of the distance to nearest and second nearest neighbor.

For privacy-safe synthetic data we expect to see about as many identical matches, and about the same distances for synthetic samples to training, as we see for synthetic samples to holdout.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ims_training` | `float | None` | Share of synthetic samples that are identical to a training sample. | `None` | | `ims_holdout` | `float | None` | Share of synthetic samples that are identical to a holdout sample. Serves as a reference for ims_training. | `None` | | `ims_trn_hol` | `float | None` | Share of training samples that are identical to a holdout sample. Serves as a reference for ims_training. | `None` | | `dcr_training` | `float | None` | Average nearest-neighbor distance between synthetic and training samples. | `None` | | `dcr_holdout` | `float | None` | Average nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for dcr_training. | `None` | | `dcr_trn_hol` | `float | None` | Average nearest-neighbor distance between training and holdout samples. Serves as a reference for dcr_training. | `None` | | `dcr_share` | `float | None` | Share of synthetic samples that are closer to a training sample than to a holdout sample. This should not be significantly larger than 50%. | `None` | | `nndr_training` | `float | None` | 10th smallest nearest-neighbor distance ratio between synthetic and training samples. | `None` | | `nndr_holdout` | `float | None` | 10th smallest nearest-neighbor distance ratio between synthetic and holdout samples. | `None` | | `nndr_trn_hol` | `float | None` | 10th smallest nearest-neighbor distance ratio between training and holdout samples. | `None` |
