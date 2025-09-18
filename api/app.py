import os
from fastapi.exceptions import RequestValidationError # pyright: ignore[reportMissingImports]
from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
import uvicorn # pyright: ignore[reportMissingImports]
from contextlib import asynccontextmanager
from api.routers import predictions, home
from scripts.config import cnfg
from api.model.load_models import load_models, MODELS
from api.utils.handlers import valid_exception_handling


HOST = cnfg["api"]["host"]
PORT = cnfg["api"]["port"]

tags_metadata = [
    {
        "name": "home",
        "description": "Entry point with hello or something",
    },
    {
        "name": "predict",
        "description": "Manage predictions: send to request to saved model, retrieve results.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    On startup load ML models into memory.
    On shutdown log a message that API is closing.
    :param app: The FastAPI application instance.
    :return: Yield control back to FastAPI.
             Execute cleanup logic after shutdown.
    """
    load_models()
    print("models loaded.")
    yield
    print("Closing API")


app = FastAPI(openapi_tags=tags_metadata, lifespan=lifespan)

app.add_exception_handler(RequestValidationError, valid_exception_handling)

app.include_router(home.router)
app.include_router(predictions.router)


if __name__ == "__main__":   # use not default 8000 port, but defined one, use 'python api_test.py' to run for debugging
    """
    Launch the FastAPI application for local debugging using `uvicorn`.
    :param HOST: Host address to run the API on. Defaults to value from config (e.g., '127.0.0.1').
    :param PORT: Port number to run the API on. Defaults to value from config (e.g., 5000).
    :return: Runs the app with `uvicorn.run()` for local development/testing.
    """
    HOST = os.getenv("HOST", HOST)  # Default to config file value ('127.0.0.1') if not set
    PORT = int(os.getenv("PORT", PORT))    # Default to config file value (5000) if not set
    uvicorn.run(app, host=HOST, port=PORT)


