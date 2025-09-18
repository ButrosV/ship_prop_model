from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]
from typing import Optional
from datetime import datetime

# TODO: consider setting all Optional PropulsionInputFull examples to None or NaN

class PropulsionInputBase(BaseModel):
    """
    Core environmental and engine-related input feature schema for propulsion modeling
    prediction retrieval with FastAPI.

    Includes meteorological (wind, wave, swell, current, air temp) and 
    engine mass flow rate data. These fields are mandatory for model inference.

    :param windSpeed: Wind speed [m/s]. Example: 4.19
    :param windDirection: Wind direction [° from true north]. Example: 205.96
    :param waveHeight: Wave height [m]. Example: 0.87
    :param waveDirection: Wave direction [° from true north]. Example: 136.04
    :param swellHeight: Swell height [m]. Example: 0.06
    :param swellDirection: Swell direction [° from true north]. Example: 55.77
    :param currentSpeed: Current speed [m/s]. Example: 0.07
    :param currentDirection: Current direction [° from true north]. Example: 159.73
    :param airTemperature: Air temperature [°C]. Example: 25.65
    :param mainEngineMassFlowRate: Main engine mass flow rate [kg/s]. Example: 0.0
    :param fuelEfficiency: Fuel efficiency (unit depends on model). Example: 0
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

    This class combines environmental, engine performance, and vessel 
    state parameters, with the following groups:
    - **Environmental**: Includes wind, wave, swell, and current data.
    - **Engine**: Covers main and auxiliary engine parameters such as fuel flow, 
      temperature, and consumption.
    - **Vessel State**: Contains vessel navigation data like latitude, 
      longitude, heading, draught, and course over ground.
    - **Operational**: Includes performance metrics such as shaft speed, torque, 
      fuel efficiency, and specific fuel consumption.
    - **Time & Conditions**: Includes timestamp, gust speed, and water salinity.

    All fields in this schema are optional except for the core environmental inputs 
    defined in `PropulsionInputBase`.
    """
    latitude: Optional[float] = Field(None, example=12.3456)
    longitude: Optional[float] = Field(None, example=78.9012)
    draught: Optional[float] = Field(None, example=6.5)
    heading: Optional[float] = Field(None, example=180.0)
    courseOverGround: Optional[float] = Field(None, example=175.0)
    status: Optional[int] = Field(None, example=1)

    mainEngineVolumeFlowRate: Optional[float] = Field(None, example=7)
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
    shaftPower: float = Field(..., example=1.575)
    speedOverGround:float = Field(..., example= 0)
    # predicting_user: ShowUser  # experiments of linking data tables, remove after testing

    # class Config:
    #     from_attributes = True  # remove after testing, SQLAlchemy specific: enable Pydantic data read from database models

