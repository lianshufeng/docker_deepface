import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.core.routes import router as routes_router
from modules.core.store import router as store_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="FaceAPI",
        description="InsightFace + FastAPI embedding service",
        version="1.0.0",
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

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
