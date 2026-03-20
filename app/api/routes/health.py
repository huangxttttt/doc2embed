from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("")
def health_check() -> dict[str, str]:
    return {
        "message": "doc2embed service is running",
        "service": settings.app_name,
        "version": settings.app_version,
    }
