from neo4j import GraphDatabase, Driver

from config import settings

_driver: Driver | None = None


def init_driver() -> None:
    global _driver
    _driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def get_driver() -> Driver:
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialised — app lifespan not started")
    return _driver
