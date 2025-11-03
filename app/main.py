from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import diffusion, exploration, fourcorner

app = FastAPI()

# new routers
app.include_router(exploration.router)
app.include_router(fourcorner.router)

# legacy router (for backwards compatibility)
app.include_router(diffusion.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080, reload=True)
