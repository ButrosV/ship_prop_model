from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]
from typing import Optional



class PropulsionInputBase(BaseModel):
    windSpeed: float = Field(..., example=4.19)
    windDirection: float = Field(..., example=205.96)
    waveHeight: float = Field(..., example=0.87)
    waveDirection: float = Field(..., example=136.04)
    swellHeight: float = Field(..., example=0.06)
    swellDirection: float = Field(..., example=55.77)
    currentSpeed: float = Field(..., example=0.07)
    currentDirection: float = Field(..., example=159.73)
    airTemperature: float = Field(..., example=25.65)
    mainEngineMassFlowRate: float = Field(..., example=0.0)
    fuelEfficiency: Optional[float] = Field(..., example=0)  # remove after testing


class PropulsionInputFull(BaseModel):
    """TODO: update with full variable set.. or use optional in PropulsionInputBase???"""
    windSpeed: float = Field(..., example=4.19)
    windDirection: float = Field(..., example=205.96)
    waveHeight: float = Field(..., example=0.87)
    waveDirection: float = Field(..., example=136.04)
    swellHeight: float = Field(..., example=0.06)
    swellDirection: float = Field(..., example=55.77)
    currentSpeed: float = Field(..., example=0.07)
    currentDirection: float = Field(..., example=159.73)
    airTemperature: float = Field(..., example=25.65)
    mainEngineMassFlowRate: float = Field(..., example=0.0)
    fuelEfficiency: Optional[float] = Field(..., example=0)


class User(BaseModel):
    """Class for user schema, test purposes - remove after testing"""
    name: str
    email: str
    password:str


class ShowUser(BaseModel):
    """Class for user schema, test purposes - remove after testing"""
    name: str
    email: str


class PropulsionOutput(BaseModel):
    shaftPower: float = Field(..., example=1.575)
    speedOverGround:float = Field(..., example= 0)
    predicting_user: ShowUser  # experiments of linking data tables, remove after testing

    class Config:
        from_attributes = True  # remove after testing, SQLAlchemy specific: enable Pydantic data read from database models


class Login(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str | None = None

