from pydantic import BaseModel, Field


class PropulsionInput(BaseModel):
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
    