import argparse

from deepface import DeepFace
from deepface.commons.logger import Logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.core.routes import router as routes_router
from modules.core.store import router as store_router

logger = Logger()


def create_app() -> FastAPI:
    app = FastAPI(
        title="DeepFace FastAPI",
        description="DeepFace embedding and face-store service powered by FastAPI.",
        version=str(DeepFace.__version__),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_router)
    app.include_router(store_router)

    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__} (FastAPI edition)!")
    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to serve the API")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
