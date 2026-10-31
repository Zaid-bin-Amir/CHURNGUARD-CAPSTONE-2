from typing import List

from pydantic import BaseModel, ConfigDict, Field

from churnguard import config


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": config.SAMPLE_CUSTOMER})

    gender: str = Field(min_length=1, max_length=40)
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str = Field(min_length=1, max_length=40)
    Dependents: str = Field(min_length=1, max_length=40)
    tenure: int = Field(ge=0, le=120)
    PhoneService: str = Field(min_length=1, max_length=40)
    MultipleLines: str = Field(min_length=1, max_length=40)
    InternetService: str = Field(min_length=1, max_length=40)
    OnlineSecurity: str = Field(min_length=1, max_length=40)
    OnlineBackup: str = Field(min_length=1, max_length=40)
    DeviceProtection: str = Field(min_length=1, max_length=40)
    TechSupport: str = Field(min_length=1, max_length=40)
    StreamingTV: str = Field(min_length=1, max_length=40)
    StreamingMovies: str = Field(min_length=1, max_length=40)
    Contract: str = Field(min_length=1, max_length=40)
    PaperlessBilling: str = Field(min_length=1, max_length=40)
    PaymentMethod: str = Field(min_length=1, max_length=60)
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


class Driver(BaseModel):
    feature: str
    value: str
    impact: float
    direction: str
    explanation: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    probability: float
    label: str
    risk_level: str
    threshold: float
    top_drivers: List[Driver]
    summary: str
    model_version: str


class BatchRequest(BaseModel):
    customers: List[CustomerInput] = Field(min_length=1, max_length=500)


class BatchItem(BaseModel):
    index: int
    probability: float
    label: str
    risk_level: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: List[BatchItem]
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    threshold: float
