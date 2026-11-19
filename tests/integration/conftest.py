import os
from collections.abc import AsyncIterator, Iterator

import pytest
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from neo4j import AsyncDriver, AsyncGraphDatabase, NotificationDisabledClassification
from testcontainers.redis import RedisContainer

try:  # testcontainers >= 4.15 moved community modules; arq pins redis<6, which caps it at 4.13
    from testcontainers.community.neo4j import Neo4jContainer
except ImportError:
    from testcontainers.neo4j import Neo4jContainer

# Keep in sync with docker-compose.yml.
NEO4J_IMAGE = os.environ.get("NEO4J_TEST_IMAGE", "neo4j:2026.10.1-community")
NEO4J_TEST_PASSWORD = "test-password"


@pytest.fixture(scope="session")
def neo4j_container() -> Iterator[Neo4jContainer]:
    with Neo4jContainer(NEO4J_IMAGE, password=NEO4J_TEST_PASSWORD) as container:
        yield container


@pytest.fixture(scope="session")
async def neo4j_driver(neo4j_container: Neo4jContainer) -> AsyncIterator[AsyncDriver]:
    uri = neo4j_container.get_connection_url()
    async with AsyncGraphDatabase.driver(
        uri,
        auth=("neo4j", NEO4J_TEST_PASSWORD),
        notifications_disabled_classifications=[NotificationDisabledClassification.UNRECOGNIZED],
    ) as driver:
        await driver.verify_connectivity()
        yield driver


async def reset_database(driver: AsyncDriver) -> None:
    """Drop all data, constraints, and indexes so each test starts from an empty schema."""
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
        constraints = await session.run("SHOW CONSTRAINTS YIELD name RETURN name")
        for name in [record["name"] async for record in constraints]:
            await session.run("DROP CONSTRAINT $name", name=name)
        indexes = await session.run(
            "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"
        )
        for name in [record["name"] async for record in indexes]:
            await session.run("DROP INDEX $name", name=name)


@pytest.fixture(scope="session")
def redis_container() -> Iterator[RedisContainer]:
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture
async def arq_redis(redis_container: RedisContainer) -> AsyncIterator[ArqRedis]:
    settings = RedisSettings(
        host=redis_container.get_container_host_ip(),
        port=int(redis_container.get_exposed_port(6379)),
    )
    pool = await create_pool(settings)
    await pool.flushdb()
    yield pool
    await pool.aclose()
