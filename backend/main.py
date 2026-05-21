import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from neo4j import Driver

import db
from agents.diagnosis_agent import DiagnosisAgent
from agents.monitor_agent import MonitorAgent
from llm.factory import warn_if_misconfigured
from mcp.tools import router as tools_router

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class ConnectionManager:
    """Thread-safe WebSocket connection registry."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active = [w for w in self._active if w is not ws]

    async def broadcast(self, message: dict) -> None:
        text = json.dumps(message)
        dead: list[WebSocket] = []
        async with self._lock:
            targets = list(self._active)
        for ws in targets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                self._active = [w for w in self._active if w not in dead]


manager        = ConnectionManager()
monitor_agent  = MonitorAgent(poll_interval_s=10)
diagnosis_agent = DiagnosisAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_driver()
    warn_if_misconfigured()
    loop = asyncio.get_event_loop()

    def on_anomaly(alert) -> None:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "anomaly_detected", "data": alert.to_dict()}),
            loop,
        )

        def _diagnose() -> None:
            try:
                diagnosis = diagnosis_agent.diagnose([alert])
                if diagnosis is None:
                    return
                monitor_agent.pause(300)
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(
                        {"type": "diagnosis_ready", "data": diagnosis.to_dict()}
                    ),
                    loop,
                )
            except Exception:
                logger.exception("on_anomaly: diagnosis failed for cell=%s", alert.cell_id)

        _executor.submit(_diagnose)

    monitor_agent.register_callback(on_anomaly)
    monitor_agent.start()

    yield

    monitor_agent.stop()
    _executor.shutdown(wait=False)
    db.close_driver()


app = FastAPI(title="AIOps Backend", version="0.1.0", lifespan=lifespan)
app.include_router(tools_router)


@app.get("/health")
def health(driver: Driver = Depends(db.get_driver)) -> dict:
    driver.verify_connectivity()
    return {"status": "ok"}


@app.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)
