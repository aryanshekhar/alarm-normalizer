from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from neo4j import Driver

import db
from mcp.tools import router as tools_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_driver()
    yield
    db.close_driver()


app = FastAPI(title="AIOps Backend", version="0.1.0", lifespan=lifespan)
app.include_router(tools_router)


@app.get("/health")
def health(driver: Driver = Depends(db.get_driver)) -> dict:
    driver.verify_connectivity()
    return {"status": "ok"}
