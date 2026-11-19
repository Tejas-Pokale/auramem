from pathlib import Path

import pytest

from memory.config import get_settings
from memory.store.migrate import MigrationError, checksum, discover, split_statements


class TestSplitStatements:
    def test_splits_on_semicolons(self) -> None:
        assert split_statements("RETURN 1; RETURN 2;") == ["RETURN 1", "RETURN 2"]

    def test_last_statement_without_semicolon(self) -> None:
        assert split_statements("RETURN 1;\nRETURN 2") == ["RETURN 1", "RETURN 2"]

    def test_strips_line_comments(self) -> None:
        text = "// header; with semicolon\nRETURN 1; // trailing\n// only comment\n"
        assert split_statements(text) == ["RETURN 1"]

    def test_strips_block_comments(self) -> None:
        text = "/* a; b */ RETURN 1 /* c */;"
        assert split_statements(text) == ["RETURN 1"]

    def test_keeps_semicolons_and_comment_markers_inside_strings(self) -> None:
        text = "RETURN 'a;b' AS x; RETURN \"//not a comment\" AS y;"
        assert split_statements(text) == ["RETURN 'a;b' AS x", 'RETURN "//not a comment" AS y']

    def test_handles_escaped_quotes(self) -> None:
        text = r"RETURN 'it\'s; fine' AS x;"
        assert split_statements(text) == [r"RETURN 'it\'s; fine' AS x"]

    def test_keeps_backtick_identifiers(self) -> None:
        text = "OPTIONS { indexConfig: { `vector.dimensions`: 768 } };"
        assert split_statements(text) == ["OPTIONS { indexConfig: { `vector.dimensions`: 768 } }"]

    def test_empty_and_comment_only(self) -> None:
        assert split_statements("") == []
        assert split_statements("// nothing\n\n  ;  ;") == []

    def test_unterminated_string_raises(self) -> None:
        with pytest.raises(MigrationError):
            split_statements("RETURN 'oops;")


class TestChecksum:
    def test_ignores_line_ending_style(self) -> None:
        assert checksum("RETURN 1;\r\nRETURN 2;\r\n") == checksum("RETURN 1;\nRETURN 2;\n")

    def test_detects_content_change(self) -> None:
        assert checksum("RETURN 1;") != checksum("RETURN 2;")


class TestDiscover:
    def test_orders_by_version(self, tmp_path: Path) -> None:
        (tmp_path / "0002_second.cypher").write_text("RETURN 2;")
        (tmp_path / "0001_first.cypher").write_text("RETURN 1;")
        (tmp_path / "README.md").write_text("ignored")
        migrations = discover(tmp_path)
        assert [(m.version, m.name) for m in migrations] == [(1, "first"), (2, "second")]
        assert migrations[0].statements == ["RETURN 1"]

    def test_rejects_badly_named_cypher_file(self, tmp_path: Path) -> None:
        (tmp_path / "constraints.cypher").write_text("RETURN 1;")
        with pytest.raises(MigrationError, match=r"constraints\.cypher"):
            discover(tmp_path)

    def test_rejects_duplicate_versions(self, tmp_path: Path) -> None:
        (tmp_path / "0001_a.cypher").write_text("RETURN 1;")
        (tmp_path / "0001_b.cypher").write_text("RETURN 1;")
        with pytest.raises(MigrationError, match="0001"):
            discover(tmp_path)

    def test_rejects_empty_migration(self, tmp_path: Path) -> None:
        (tmp_path / "0001_empty.cypher").write_text("// nothing here\n")
        with pytest.raises(MigrationError, match="no statements"):
            discover(tmp_path)


def test_project_migrations_parse() -> None:
    migrations = discover(get_settings().migrations_dir)
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    statements = [s for m in migrations for s in m.statements]
    for s in statements:
        idempotent = (s.startswith("CREATE ") and "IF NOT EXISTS" in s) or (
            s.startswith("DROP ") and s.endswith("IF EXISTS")
        )
        assert idempotent, s
