import os
from fastapi import FastAPI, status, Request, Response # pyright: ignore[reportMissingImports]
from fastapi.responses import HTMLResponse # pyright: ignore[reportMissingImports]
import uvicorn # pyright: ignore[reportMissingImports]
from api.schema import PropulsionInputBase, PropulsionInputFull, PropulsionOutput
from api.routers import predictions, home
from scripts.model.get_model import load_train_model
from scripts.config import cnfg

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

app = FastAPI(openapi_tags=tags_metadata)

app.include_router(home.router)
app.include_router(predictions.router)


if __name__ == "__main__":   # use not default 8000 port, but defined one, use 'python api_test.py' to run for debugging
    HOST = os.getenv("HOST", HOST)  # Default to config file value ('127.0.0.1') if not set
    PORT = int(os.getenv("PORT", PORT))    # Default to config file value (5000) if not set
    uvicorn.run(app, host=HOST, port=PORT)
