from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from neo4j import GraphDatabase, Driver

from config import settings
from mcp.tools import router as tools_router


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


_driver: Driver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver
    _driver = get_neo4j_driver()
    yield
    _driver.close()


app = FastAPI(title="AIOps Backend", version="0.1.0", lifespan=lifespan)
app.include_router(tools_router)


@app.get("/health")
def health(driver: Driver = Depends(get_neo4j_driver)) -> dict:
    driver.verify_connectivity()
    return {"status": "ok"}
