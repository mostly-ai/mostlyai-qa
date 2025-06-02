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

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True, validate_assignment=True)


class Accuracy(CustomBaseModel):
    """
    Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional
    marginal distributions.

    1. **Univariate Accuracy**: The accuracy of the univariate distributions for all target columns.
    2. **Bivariate Accuracy**: The accuracy of all pair-wise distributions for target columns, as well as for target
    columns with respect to the context columns.
    3. **Trivariate Accuracy**: The accuracy of all three-way distributions for target columns.
    4. **Coherence Accuracy**: The accuracy of the auto-correlation for all target columns.

    Accuracy is defined as 100% - [Total Variation Distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) (TVD),
    whereas TVD is half the sum of the absolute differences of the relative frequencies of the corresponding
    distributions.

    These accuracies are calculated for all discretized univariate, and bivariate distributions. In case of sequential
    data, also for all coherence distributions. Overall metrics are then calculated as the average across these
    accuracies.

    All metrics can be compared against a theoretical maximum accuracy, which is calculated for a same-sized holdout.
    The accuracy metrics shall be as close as possible to the theoretical maximum, but not significantly higher, as
    this would indicate overfitting.
    """

    overall: float | None = Field(
        default=None,
        description="Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence.",
        ge=0.0,
        le=1.0,
    )
    univariate: float | None = Field(
        default=None,
        description="Average accuracy of discretized univariate distributions.",
        ge=0.0,
        le=1.0,
    )
    bivariate: float | None = Field(
        default=None,
        description="Average accuracy of discretized bivariate distributions.",
        ge=0.0,
        le=1.0,
    )
    trivariate: float | None = Field(
        default=None,
        description="Average accuracy of discretized trivariate distributions.",
        ge=0.0,
        le=1.0,
    )
    coherence: float | None = Field(
        default=None,
        description="Average accuracy of discretized coherence distributions. Only applicable for sequential data.",
        ge=0.0,
        le=1.0,
    )
    overall_max: float | None = Field(
        default=None,
        alias="overallMax",
        description="Expected overall accuracy of a same-sized holdout. Serves as a reference for `overall`.",
        ge=0.0,
        le=1.0,
    )
    univariate_max: float | None = Field(
        default=None,
        alias="univariateMax",
        description="Expected univariate accuracy of a same-sized holdout. Serves as a reference for `univariate`.",
        ge=0.0,
        le=1.0,
    )
    bivariate_max: float | None = Field(
        default=None,
        alias="bivariateMax",
        description="Expected bivariate accuracy of a same-sized holdout. Serves as a reference for `bivariate`.",
        ge=0.0,
        le=1.0,
    )
    trivariate_max: float | None = Field(
        default=None,
        alias="trivariateMax",
        description="Expected trivariate accuracy of a same-sized holdout. Serves as a reference for `trivariate`.",
        ge=0.0,
        le=1.0,
    )
    coherence_max: float | None = Field(
        default=None,
        alias="coherenceMax",
        description="Expected coherence accuracy of a same-sized holdout. Serves as a reference for `coherence`.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("*", mode="after")
    def trim_metric_precision(cls, value):
        precision = 7
        return round(value, precision) if value is not None else None


class Similarity(CustomBaseModel):
    """
    Metrics regarding the similarity of the full joint distributions of samples within an embedding space.

    1. **Cosine Similarity**: The cosine similarity between the centroids of synthetic and training samples.
    2. **Discriminator AUC**: The AUC of a discriminative model to distinguish between synthetic and training samples.

    The Model2Vec model [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) is
    used to compute the embeddings of a string-ified representation of individual records. In case of sequential data
    the records, that belong to the same group, are being concatenated. We then calculate the cosine similarity
    between the centroids of the provided datasets within the embedding space.

    Again, we expect the similarity metrics to be as close as possible to 1, but not significantly higher than what is
    measured for the holdout data, as this would again indicate overfitting.

    In addition, a discriminative ML model is trained to distinguish between training and synthetic samples. The
    ability of this model to distinguish between training and synthetic samples is measured by the AUC score. For
    synthetic data to be considered realistic, the AUC score should be close to 0.5, which indicates that the synthetic
    data is indistinguishable from the training data.
    """

    cosine_similarity_training_synthetic: float | None = Field(
        default=None,
        alias="cosineSimilarityTrainingSynthetic",
        description="Cosine similarity between training and synthetic centroids.",
        ge=-1.0,
        le=1.0,
    )
    cosine_similarity_training_holdout: float | None = Field(
        default=None,
        alias="cosineSimilarityTrainingHoldout",
        description="Cosine similarity between training and holdout centroids. Serves as a reference for "
        "`cosine_similarity_training_synthetic`.",
        ge=-1.0,
        le=1.0,
    )
    discriminator_auc_training_synthetic: float | None = Field(
        default=None,
        alias="discriminatorAUCTrainingSynthetic",
        description="Cross-validated AUC of a discriminative model to distinguish between training and synthetic "
        "samples.",
        ge=0.0,
        le=1.0,
    )
    discriminator_auc_training_holdout: float | None = Field(
        default=None,
        alias="discriminatorAUCTrainingHoldout",
        description="Cross-validated AUC of a discriminative model to distinguish between training and holdout "
        "samples. Serves as a reference for `discriminator_auc_training_synthetic`.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("*", mode="after")
    def trim_metric_precision(cls, value, info):
        precision = 7 if "cosine" in info.field_name else 3
        return round(value, precision) if value is not None else None


class Distances(CustomBaseModel):
    """
    Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numerically
    encoded space. Useful for assessing the novelty / privacy of synthetic data.

    The provided data is first down-sampled, so that the number of samples match across all datasets. Note, that for
    an optimal sensitivity of this privacy assessment it is recommended to use a 50/50 split between training and
    holdout data, and then generate synthetic data of the same size.

    The numerical encodings of these samples are then computed, and the nearest neighbor distances are calculated for each
    synthetic sample to the training and holdout samples. Based on these nearest neighbor distances the following
    metrics are calculated:
    - Identical Match Share (IMS): The share of synthetic samples that are identical to a training or holdout sample.
    - Distance to Closest Record (DCR): The average distance of synthetic to training or holdout samples.
    - Nearest Neighbor Distance Ratio (NNDR): The 10-th smallest ratio of the distance to nearest and second nearest neighbor.

    For privacy-safe synthetic data we expect to see about as many identical matches, and about the same distances
    for synthetic samples to training, as we see for synthetic samples to holdout.
    """

    ims_training: float | None = Field(
        default=None,
        alias="imsTraining",
        description="Share of synthetic samples that are identical to a training sample.",
        ge=0.0,
        le=1.0,
    )
    ims_holdout: float | None = Field(
        default=None,
        alias="imsHoldout",
        description="Share of synthetic samples that are identical to a holdout sample. Serves as a reference for "
        "`ims_training`.",
        ge=0.0,
        le=1.0,
    )
    ims_trn_hol: float | None = Field(
        default=None,
        alias="imsTrnHol",
        description="Share of training samples that are identical to a holdout sample. Serves as a reference for "
        "`ims_training`.",
        ge=0.0,
        le=1.0,
    )
    dcr_training: float | None = Field(
        default=None,
        alias="dcrTraining",
        description="Average nearest-neighbor distance between synthetic and training samples.",
        ge=0.0,
    )
    dcr_holdout: float | None = Field(
        default=None,
        alias="dcrHoldout",
        description="Average nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for `dcr_training`.",
        ge=0.0,
    )
    dcr_trn_hol: float | None = Field(
        default=None,
        alias="dcrTrnHol",
        description="Average nearest-neighbor distance between training and holdout samples. Serves as a reference for `dcr_training`.",
        ge=0.0,
    )
    dcr_share: float | None = Field(
        default=None,
        alias="dcrShare",
        description="Share of synthetic samples that are closer to a training sample than to a holdout sample. This "
        "should not be significantly larger than 50%.",
        ge=0.0,
        le=1.0,
    )
    nndr_training: float | None = Field(
        default=None,
        alias="nndrTraining",
        description="10th smallest nearest-neighbor distance ratio between synthetic and training samples.",
        ge=0.0,
    )
    nndr_holdout: float | None = Field(
        default=None,
        alias="nndrHoldout",
        description="10th smallest nearest-neighbor distance ratio between synthetic and holdout samples.",
        ge=0.0,
    )
    nndr_trn_hol: float | None = Field(
        default=None,
        alias="nndrTrnHol",
        description="10th smallest nearest-neighbor distance ratio between training and holdout samples.",
        ge=0.0,
    )

    @field_validator("*", mode="after")
    def trim_metric_precision(cls, value, info):
        precision = 12 if "nndr" in info.field_name else 3
        return round(value, precision) if value is not None else None


class ModelMetrics(CustomBaseModel):
    """
    Metrics regarding the quality of synthetic data, measured in terms of accuracy, similarity, and distances.

    1. **Accuracy**: Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower
    dimensional marginal distributions.
    2. **Similarity**: Metrics regarding the similarity of the full joint distributions of samples within an embedding
    space.
    3. **Distances**: Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples
    in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data.

    The quality of synthetic data is assessed by comparing these metrics to the same metrics of a holdout dataset.
    The holdout dataset is a subset of the original training data, that was not used for training the synthetic data
    generator. The metrics of the synthetic data should be as close as possible to the metrics of the holdout data.
    """

    accuracy: Accuracy | None = Field(
        default=None,
        description="Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower "
        "dimensional marginal distributions.",
    )
    similarity: Similarity | None = Field(
        default=None,
        description="Metrics regarding the similarity of the full joint distributions of samples within an embedding "
        "space.",
    )
    distances: Distances | None = Field(
        default=None,
        description="Metrics regarding the nearest neighbor distances between training, holdout, and synthetic "
        "samples in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data.",
    )
