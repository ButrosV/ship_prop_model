from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]
from typing import Optional
from datetime import datetime


class PropulsionInputBase(BaseModel):
    """
    Core environmental and engine-related input feature schema for propulsion modeling
    prediction retrieval with FastAPI.
    Includes meteorological (wind, wave, swell, current, air temp) and 
    engine mass flow rate data. These fields are mandatory for model inference.
    All fields are required and must be provided by the model's prediction output.
    Example values are included to illustrate expected data types and units.
    """
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


class PropulsionInputFull(PropulsionInputBase):
    """
    Extended propulsion input schema including optional environmental, 
    engine, and vessel state features for full model input for models 
    trained on full feature set.
    All fields in this schema are optional except for the core environmental inputs 
    defined in `PropulsionInputBase`. Optional fields default to None/NaN.
    """
    latitude: Optional[float] = Field(None)
    longitude: Optional[float] = Field(None)
    draught: Optional[float] = Field(None)
    heading: Optional[float] = Field(None)
    courseOverGround: Optional[float] = Field(None)
    status: Optional[int] = Field(None)

    mainEngineVolumeFlowRate: Optional[float] = Field(None)
    mainEngineDensity: Optional[float] = Field(None)
    mainEngineTemperature: Optional[float] = Field(None)
    mainEngineFuelConsumed: Optional[float] = Field(None)
    
    auxEngineMassFlowRate: Optional[float] = Field(None)
    auxEngineVolumeFlowRate: Optional[float] = Field(None)
    auxEngineDensity: Optional[float] = Field(None)
    auxEngineTemperature: Optional[float] = Field(None)
    
    shaftSpeed: Optional[float] = Field(None)
    shaftTorque: Optional[float] = Field(None)
    specificFuelOilConsumption: Optional[float] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    fuelEfficiency: Optional[float] = Field(None)
    
    gust: Optional[float] = Field(None)
    swellPeriod: Optional[float] = Field(None)
    wavePeriod: Optional[float] = Field(None)

    windWaveDirection: Optional[float] = Field(None)
    windWaveHeight: Optional[float] = Field(None)
    windWavePeriod: Optional[float] = Field(None)
    
    salinity: Optional[float] = Field(None)
    seaLevel: Optional[float] = Field(None)


class PropulsionOutput(BaseModel):
    """
    Propulsion output schema representing expected predicted values.
    All fields are required and must be provided by the model's prediction output.
    Example values are included to illustrate expected data types and units.
    """
    shaftPower: float = Field(..., example=1.575)
    speedOverGround:float = Field(..., example= 0)
