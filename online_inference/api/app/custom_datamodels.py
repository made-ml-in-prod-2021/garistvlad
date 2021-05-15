from typing import List

from fastapi import HTTPException
import pandas as pd
from pydantic import BaseModel, validator


class HeartRequestModel(BaseModel):
    # binary features
    sex: int
    fbs: int
    exang: int
    # categorical features
    cp: int
    restecg: int
    slope: int
    ca: int
    thal: int
    # numerical features
    age: int
    trestbps: int
    chol: int
    thalach: int
    oldpeak: float

    def get_fields(self) -> List[str]:
        """Get list of field names for the Model"""
        list_of_fields = list(self.dict().keys())
        return list_of_fields

    def to_pandas(self) -> pd.DataFrame:
        """Transform BaseModel to row of pandas.DataFrame"""
        data = pd.DataFrame.from_dict([self.dict()], orient='columns')
        return data

    @validator('sex', 'fbs', 'exang')
    def binary_features_values(cls, v):
        if v not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail="Binary feature should be equal to 0 or 1."
            )
        return v

    @validator("cp")
    def range_of_values_cp(cls, v):
        if v not in (0, 1, 2, 3):
            raise HTTPException(
                status_code=400,
                detail="Value of `cp` should be one of: [0, 1, 2, 3]"
            )
        return v

    @validator("restecg")
    def range_of_values_restecg(cls, v):
        if v not in (0, 1, 2):
            raise HTTPException(
                status_code=400,
                detail="Value of `restecg` should be one of: [0, 1, 2]"
            )
        return v

    @validator("slope")
    def range_of_values_slope(cls, v):
        if v not in (0, 1, 2):
            raise HTTPException(
                status_code=400,
                detail="Value of `slope` should be one of: [0, 1, 2]"
            )
        return v

    @validator("ca")
    def range_of_values_ca(cls, v):
        if v not in (0, 1, 2, 3, 4):
            raise HTTPException(
                status_code=400,
                detail="Value of `ca` should be one of: [0, 1, 2, 3, 4]"
            )
        return v

    @validator("thal")
    def range_of_values_thal(cls, v):
        if v not in (0, 1, 2, 3):
            raise HTTPException(
                status_code=400,
                detail="Value of `thal` should be one of: [0, 1, 2, 3]"
            )
        return v

    @validator("age")
    def incorrect_value_age(cls, v):
        if not (0 < v <= 150):
            raise HTTPException(
                status_code=400,
                detail=f"Incorrect `age` value: {v}"
            )
        return v

    @validator("trestbps", "thalach")
    def incorrect_value_blood_pressure(cls, v):
        if not (50 <= v <= 220):
            raise HTTPException(
                status_code=400,
                detail=f"Incorrect value for blood pressure: {v}"
            )
        return v


class HeartResponseModel(BaseModel):
    predicted_probability: float

    @validator("predicted_probability")
    def range_of_probability(cls, v):
        if not (0 <= v <= 1):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Probability value should belong to interval [0, 1]. "
                    f"But your value is: {v}"
                )
            )
        return v

    def get_label(self, threshold=0.5) -> int:
        """Get label (0/1) prediction based on predefined threshold"""
        return int(self.predicted_probability > threshold)
