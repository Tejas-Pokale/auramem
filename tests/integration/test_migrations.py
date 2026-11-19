from dataclasses import replace

import pytest
from neo4j import AsyncDriver

from memory.config import get_settings
from memory.store.migrate import (
    MigrationError,
    apply_migrations,
    discover,
    schema_snapshot,
)
from tests.integration.conftest import reset_database

pytestmark = pytest.mark.integration

DB = "neo4j"
TIMEOUT = 120

EXPECTED_CONSTRAINTS = {
    "user_user_id": ("User", ["user_id"]),
    "session_session_id": ("Session", ["session_id"]),
    "turn_turn_id": ("Turn", ["turn_id"]),
    "fact_fact_id": ("Fact", ["fact_id"]),
    "entity_entity_id": ("Entity", ["entity_id"]),
    "observation_log_user_id_unique": ("ObservationLog", ["user_id"]),
    "turn_session_seq": ("Turn", ["session_id", "seq"]),
    "fact_dedup_key": ("Fact", ["dedup_key"]),
}

EXPECTED_VECTOR = {
    "turn_vec": ("Turn", ["embedding", "user_id", "created_at"]),
    "fact_vec": ("Fact", ["embedding", "user_id", "is_latest", "is_hypothetical"]),
    "entity_vec": ("Entity", ["embedding", "user_id"]),
}

EXPECTED_FULLTEXT = {
    "turn_text": ("Turn", ["text", "user_key"]),
    "fact_text": ("Fact", ["text", "aliases_text", "user_key", "flags_text"]),
    "entity_text": ("Entity", ["name", "aliases_text", "user_key"]),
}

EXPECTED_RANGE = {
    "fact_user_id": ("Fact", ["user_id"]),
    "fact_start": ("Fact", ["start"]),
    "fact_end": ("Fact", ["end"]),
    "fact_mentioned_at": ("Fact", ["mentioned_at"]),
    "fact_attribute_key": ("Fact", ["attribute_key"]),
    "turn_created_at": ("Turn", ["created_at"]),
}


@pytest.fixture
async def fresh_db(neo4j_driver: AsyncDriver) -> AsyncDriver:
    await reset_database(neo4j_driver)
    return neo4j_driver


async def test_creates_all_constraints_and_indexes(fresh_db: AsyncDriver) -> None:
    migrations = discover(get_settings().migrations_dir)
    applied = await apply_migrations(
        fresh_db, migrations, database=DB, index_timeout_seconds=TIMEOUT
    )
    assert applied == [m.version for m in migrations]

    snap = await schema_snapshot(fresh_db, database=DB)
    assert snap.applied_versions == applied

    for name, expected in EXPECTED_CONSTRAINTS.items():
        assert snap.constraints[name] == expected, name

    for expected_type, expected_indexes in [
        ("VECTOR", EXPECTED_VECTOR),
        ("FULLTEXT", EXPECTED_FULLTEXT),
        ("RANGE", EXPECTED_RANGE),
    ]:
        for name, (label, props) in expected_indexes.items():
            idx = snap.indexes[name]
            assert (idx.type, idx.label, idx.properties) == (expected_type, label, props), name

    not_online = {i.name: i.state for i in snap.indexes.values() if i.state != "ONLINE"}
    assert not_online == {}


async def test_rerun_is_noop(fresh_db: AsyncDriver) -> None:
    migrations = discover(get_settings().migrations_dir)
    await apply_migrations(fresh_db, migrations, database=DB, index_timeout_seconds=TIMEOUT)
    before = await schema_snapshot(fresh_db, database=DB)

    again = await apply_migrations(fresh_db, migrations, database=DB, index_timeout_seconds=TIMEOUT)

    assert again == []
    after = await schema_snapshot(fresh_db, database=DB)
    assert set(after.indexes) == set(before.indexes)
    assert after.constraints == before.constraints


async def test_applies_only_pending(fresh_db: AsyncDriver) -> None:
    migrations = discover(get_settings().migrations_dir)
    first = await apply_migrations(
        fresh_db, migrations[:1], database=DB, index_timeout_seconds=TIMEOUT
    )
    rest = await apply_migrations(fresh_db, migrations, database=DB, index_timeout_seconds=TIMEOUT)
    assert first == [migrations[0].version]
    assert rest == [m.version for m in migrations[1:]]


async def test_edited_migration_is_rejected(fresh_db: AsyncDriver) -> None:
    migrations = discover(get_settings().migrations_dir)
    await apply_migrations(fresh_db, migrations, database=DB, index_timeout_seconds=TIMEOUT)

    tampered = [replace(migrations[0], checksum="0" * 64), *migrations[1:]]
    with pytest.raises(MigrationError, match="edited after it was applied"):
        await apply_migrations(fresh_db, tampered, database=DB, index_timeout_seconds=TIMEOUT)
