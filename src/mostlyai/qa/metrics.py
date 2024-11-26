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

from pydantic import BaseModel, Field, field_validator, ConfigDict


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class Accuracy(CustomBaseModel):
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
    coherence_max: float | None = Field(
        default=None,
        alias="coherenceMax",
        description="Expected coherence accuracy of a same-sized holdout. Serves as a reference for `coherence`.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value):
        precision = 3
        return round(value, precision) if value is not None else None


class Similarity(CustomBaseModel):
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
        alias="discriminatorAucTrainingSynthetic",
        description="Cross-validated AUC of a discriminative model to distinguish between training and synthetic "
        "samples.",
        ge=0.0,
        le=1.0,
    )
    discriminator_auc_training_holdout: float | None = Field(
        default=None,
        alias="discriminatorAucTrainingHoldout",
        description="Cross-validated AUC of a discriminative model to distinguish between training and holdout "
        "samples. Serves as a reference for `discriminator_auc_training_synthetic`.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value, info):
        precision = 7 if "cosine" in info.field_name else 3
        return round(value, precision) if value is not None else None


class Distances(CustomBaseModel):
    ims_training: float | None = Field(
        default=None,
        alias="imsTraining",
        description="Share of synthetic samples that are identical to a training sample.",
        ge=0.0,
    )
    ims_holdout: float | None = Field(
        default=None,
        alias="imsHoldout",
        description="Share of synthetic samples that are identical to a holdout sample. Serves as a reference for "
        "`ims_training`.",
        ge=0.0,
    )
    dcr_training: float | None = Field(
        default=None,
        alias="dcrTraining",
        description="Average L2 nearest-neighbor distance between synthetic and training samples.",
        ge=0.0,
    )
    dcr_holdout: float | None = Field(
        default=None,
        alias="dcrHoldout",
        description="Average L2 nearest-neighbor distance between synthetic and holdout samples. Serves as a "
        "reference for `dcr_training`.",
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

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value):
        precision = 3
        return round(value, precision) if value is not None else None


class Metrics(CustomBaseModel):
    accuracy: Accuracy | None = Field(default=None, description="Metrics related to accuracy of synthetic data.")
    similarity: Similarity | None = Field(
        default=None, description="Metrics related to similarity between distributions in an embedding space."
    )
    distances: Distances | None = Field(
        default=None,
        description="Metrics related to nearest neighbor distances between training, holdout, and synthetic samples "
        "in an embedding space.",
    )
