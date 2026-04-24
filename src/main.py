from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.api.routers import lstm_router


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(lstm_router.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1")