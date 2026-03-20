from fastapi import APIRouter

from app.api.routes.database import router as database_router
from app.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(database_router, prefix="/database", tags=["database"])
