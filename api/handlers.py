from fastapi.exceptions import RequestValidationError # pyright: ignore[reportMissingImports]
from fastapi import Request # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse # pyright: ignore[reportMissingImports]

async def valid_exception_handling(request: Request, exc: RequestValidationError):
    """
    Handle input validation errors raised by FastAPI during request parsing.
    :param request: FastAPI request object containing metadata about the failed request.
    :param exc: The raised RequestValidationError exception containing error details.
    :return: JSONResponse with 422 status code and message describing the validation issue.
    """
    print(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422,
                        content={"detail": exc.errors(),
                                 "message": "Not all mandatory fields/features \
                                    provided for basic prediction model with 'limited feature set'."}
                                 )
