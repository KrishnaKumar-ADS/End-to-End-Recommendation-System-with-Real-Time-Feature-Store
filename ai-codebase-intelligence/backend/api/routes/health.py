from fastapi import APIRouter
from graphs.neo4j_client import ping as neo4j_ping

router = APIRouter()

@router.get("/health")
async def health_check():
    from reasoning.llm_router import get_available_providers
    available_providers = [p.value for p in get_available_providers()]

    neo4j_ok = neo4j_ping()

    return {
        "status": "ok",
        "version": "0.1.0",
        "llm_providers_available": available_providers,
        "databases": {
            "neo4j": "connected" if neo4j_ok else "unavailable",
        },
    }