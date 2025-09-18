from fastapi import APIRouter, Request # pyright: ignore[reportMissingImports]
from fastapi.responses import HTMLResponse # pyright: ignore[reportMissingImports]

router = APIRouter(
    prefix = "",  # add prefix to each router
    tags=["home"]  # add tag to each router, no need to add individual tags to each router
)

@router.get('/', tags=["home"])  # .get/.post/.put, .etc - operation, inside ' ' - endpoint/path, @app - path operation decorator
def home(request: Request):
    """Root endpoint that provides a welcome message and API usage instructions.

    Returns a brief introduction to the Ship Propulsion Prediction API and 
    provides links to explore predictions and modify feature values.
    :return: A welcome message with guidance for using the API.
    """
    url = str(request.base_url)
    text = f"""Welcome to Ship Propulsion Prediction API.<br><br>
    - To see predictions with default values, go to '{url}predictions'.<br><br>
    - To modify feature values for predictions, use '{url}docs'.
    """
    return HTMLResponse(content=text)
