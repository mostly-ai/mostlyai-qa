from pydantic import BaseModel, Field, field_validator


class AccuracyMetrics(BaseModel):
    overall: float | None = Field(
        None, description="Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence."
    )
    univariate: float | None = Field(None, description="Average accuracy of discretized univariate distributions.")
    bivariate: float | None = Field(None, description="Average accuracy of discretized bivariate distributions.")
    coherence: float | None = Field(
        None,
        description="Average accuracy of discretized coherence distributions. Only applicable for sequential data.",
    )
    overall_max: float | None = Field(
        None, description="Expected overall accuracy of a same-sized holdout. Serves as a reference for `overall`."
    )
    univariate_max: float | None = Field(
        None,
        description="Expected univariate accuracy of a same-sized holdout. Serves as a reference for `univariate`.",
    )
    bivariate_max: float | None = Field(
        None, description="Expected bivariate accuracy of a same-sized holdout. Serves as a reference for `bivariate`."
    )
    coherence_max: float | None = Field(
        None, description="Expected coherence accuracy of a same-sized holdout. Serves as a reference for `coherence`."
    )

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value):
        precision = 3
        return round(value, precision) if value is not None else None


class SimilarityMetrics(BaseModel):
    cosine_similarity_training_synthetic: float | None = Field(
        None, description="Cosine similarity between training and synthetic centroids."
    )
    cosine_similarity_training_holdout: float | None = Field(
        None,
        description="Cosine similarity between training and holdout centroids. Serves as a reference for `cosine_similarity_training_synthetic`.",
    )
    discriminator_auc_training_synthetic: float | None = Field(
        None,
        description="Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples.",
    )
    discriminator_auc_training_holdout: float | None = Field(
        None,
        description="Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as a reference for `discriminator_auc_training_synthetic`.",
    )

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value, info):
        precision = 7 if "cosine" in info.field_name else 3
        return round(value, precision) if value is not None else None


class DistanceMetrics(BaseModel):
    ims_training: float | None = Field(
        None, description="Share of synthetic samples that are identical to a training sample."
    )
    ims_holdout: float | None = Field(
        None,
        description="Share of synthetic samples that are identical to a holdout sample. Serves as a reference for `ims_training`.",
    )
    dcr_training: float | None = Field(
        None, description="Average L2 nearest-neighbor distance between synthetic and training samples."
    )
    dcr_holdout: float | None = Field(
        None,
        description="Average L2 nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for `dcr_training`.",
    )
    dcr_share: float | None = Field(
        None,
        description="Share of synthetic samples that are closer to a training sample than to a holdout sample. This should not be significantly larger than 50%.",
    )

    @field_validator("*", mode="before")
    def trim_metric_precision(cls, value):
        precision = 3
        return round(value, precision) if value is not None else None


class Metrics(BaseModel):
    accuracy: AccuracyMetrics = Field(..., description="Metrics related to accuracy of synthetic data.")
    similarity: SimilarityMetrics = Field(
        ..., description="Metrics related to similarity between distributions in an embedding space."
    )
    distances: DistanceMetrics = Field(
        ...,
        description="Metrics related to nearest neighbor distances between training, holdout, and synthetic samples in an embedding space.",
    )
