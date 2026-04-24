from fastapi import APIRouter
from .routers import lstm_router

router = APIRouter()

router.include_router(lstm_router.router)