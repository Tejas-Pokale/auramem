"""Apply numbered ``.cypher`` migrations idempotently.

Usage: ``uv run python -m memory.store.migrate [--status]``

Files in ``store/migrations/`` are named ``NNNN_description.cypher`` and applied in
version order. Each applied file is recorded as a ``(:SchemaMigration)`` node with a
checksum, so re-running is a no-op and editing an applied file is an error (add a new
migration instead). Statements run one per auto-commit transaction because Neo4j does
not allow schema and data changes in the same transaction.
"""

import argparse
import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from neo4j import AsyncDriver, AsyncGraphDatabase

from memory.config import Settings, get_settings

log = logging.getLogger(__name__)

_FILENAME = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.cypher$")

_ENSURE_TRACKING = """
CREATE CONSTRAINT schema_migration_version IF NOT EXISTS
FOR (m:SchemaMigration) REQUIRE m.version IS UNIQUE
"""
_APPLIED = "MATCH (m:SchemaMigration) RETURN m.version AS version, m.checksum AS checksum"
_RECORD = """
CREATE (:SchemaMigration {
  version: $version, name: $name, checksum: $checksum, applied_at: datetime()
})
"""
_AWAIT_INDEXES = "CALL db.awaitIndexes($timeout)"
_SHOW_CONSTRAINTS = "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties RETURN *"
_SHOW_INDEXES = """
SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties, owningConstraint
WHERE type <> 'LOOKUP'
RETURN *
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    checksum: str
    statements: list[str]


@dataclass(frozen=True)
class IndexInfo:
    name: str
    type: str  # RANGE, VECTOR, FULLTEXT, ...
    state: str  # ONLINE, POPULATING, FAILED
    label: str
    properties: list[str]
    owning_constraint: str | None


@dataclass(frozen=True)
class SchemaSnapshot:
    applied_versions: list[int]
    constraints: dict[str, tuple[str, list[str]]]  # name -> (label, properties)
    indexes: dict[str, IndexInfo]


def checksum(text: str) -> str:
    """SHA-256 of the file text with line endings normalized (git autocrlf safe)."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def split_statements(text: str) -> list[str]:
    """Split Cypher on top-level ``;``, dropping ``//`` and ``/* */`` comments.

    Quotes (``'``, ``"``) and backtick identifiers are respected, so semicolons and
    comment markers inside them are kept.
    """
    statements: list[str] = []
    current: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    while i < n:
        ch = text[i]
        if quote:
            current.append(ch)
            if ch == "\\" and quote != "`" and i + 1 < n:
                current.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
        elif ch in "'\"`":
            quote = ch
            current.append(ch)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                raise MigrationError("unterminated block comment")
            current.append(" ")
            i = end + 2
        elif ch == ";":
            statements.append("".join(current))
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    if quote:
        raise MigrationError(f"unterminated {quote} quote")
    statements.append("".join(current))
    return [s.strip() for s in statements if s.strip()]


def discover(directory: Path) -> list[Migration]:
    migrations: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.cypher")):
        match = _FILENAME.match(path.name)
        if not match:
            raise MigrationError(f"{path.name}: expected name like 0001_description.cypher")
        version = int(match["version"])
        if version in migrations:
            raise MigrationError(f"duplicate migration version {match['version']}")
        text = path.read_text(encoding="utf-8")
        statements = split_statements(text)
        if not statements:
            raise MigrationError(f"{path.name}: no statements")
        migrations[version] = Migration(version, match["name"], checksum(text), statements)
    return [migrations[v] for v in sorted(migrations)]


async def apply_migrations(
    driver: AsyncDriver,
    migrations: list[Migration],
    *,
    database: str,
    index_timeout_seconds: int,
) -> list[int]:
    """Apply pending migrations in order and wait for indexes. Returns applied versions."""
    async with driver.session(database=database) as session:
        await session.run(_ENSURE_TRACKING)
        result = await session.run(_APPLIED)
        applied = {record["version"]: record["checksum"] async for record in result}

        newly_applied: list[int] = []
        for migration in migrations:
            if migration.version in applied:
                if applied[migration.version] != migration.checksum:
                    raise MigrationError(
                        f"migration {migration.version:04d}_{migration.name} was edited after "
                        "it was applied; add a new migration instead"
                    )
                continue
            log.info("applying migration %04d_%s", migration.version, migration.name)
            for statement in migration.statements:
                await session.run(statement)
            await session.run(
                _RECORD,
                version=migration.version,
                name=migration.name,
                checksum=migration.checksum,
            )
            newly_applied.append(migration.version)

        await session.run(_AWAIT_INDEXES, timeout=index_timeout_seconds)
    return newly_applied


async def schema_snapshot(driver: AsyncDriver, *, database: str) -> SchemaSnapshot:
    """Applied migration versions plus every constraint and non-lookup index."""
    async with driver.session(database=database) as session:
        applied = [r["version"] async for r in await session.run(_APPLIED)]
        constraints = {
            r["name"]: (r["labelsOrTypes"][0], list(r["properties"]))
            async for r in await session.run(_SHOW_CONSTRAINTS)
        }
        indexes = {
            r["name"]: IndexInfo(
                name=r["name"],
                type=r["type"],
                state=r["state"],
                label=r["labelsOrTypes"][0],
                properties=list(r["properties"]),
                owning_constraint=r["owningConstraint"],
            )
            async for r in await session.run(_SHOW_INDEXES)
        }
    return SchemaSnapshot(sorted(applied), constraints, indexes)


def _driver(settings: Settings) -> AsyncDriver:
    auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
    # Schema DDL and the first tracking-node read trigger harmless "property does not
    # exist" notices on an empty database.
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=auth, notifications_min_severity="OFF"
    )


async def migrate(settings: Settings) -> list[int]:
    migrations = discover(settings.migrations_dir)
    async with _driver(settings) as driver:
        return await apply_migrations(
            driver,
            migrations,
            database=settings.neo4j_database,
            index_timeout_seconds=settings.migration_index_timeout_seconds,
        )


async def status(settings: Settings) -> SchemaSnapshot:
    async with _driver(settings) as driver:
        return await schema_snapshot(driver, database=settings.neo4j_database)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply Neo4j schema migrations.")
    parser.add_argument("--status", action="store_true", help="show schema, apply nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    if args.status:
        snap = asyncio.run(status(settings))
        log.info("applied migrations: %s", snap.applied_versions)
        for idx in sorted(snap.indexes.values(), key=lambda i: (i.type, i.name)):
            log.info(
                "%-9s %-9s %-28s %s%s", idx.type, idx.state, idx.name, idx.label, idx.properties
            )
        return

    applied = asyncio.run(migrate(settings))
    if applied:
        log.info("applied %d migration(s): %s", len(applied), applied)
    else:
        log.info("schema is up to date")


if __name__ == "__main__":
    main()
