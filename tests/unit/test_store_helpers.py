import hashlib
import re

from memory.store.neo4j import fulltext_query, lucene_terms, user_key


def test_user_key_is_sha256_hex() -> None:
    assert user_key("alice") == hashlib.sha256(b"alice").hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", user_key("weird id: with spaces/+-"))


def test_user_key_differs_per_user() -> None:
    assert user_key("alice") != user_key("alice2")


class TestLuceneTerms:
    def test_lowercases_to_disable_boolean_operators(self) -> None:
        assert lucene_terms("cats AND dogs OR NOT birds") == [
            "cats",
            "and",
            "dogs",
            "or",
            "not",
            "birds",
        ]

    def test_escapes_special_characters(self) -> None:
        assert lucene_terms("user_key:abc") == [r"user_key\:abc"]
        assert lucene_terms("C++ (v2) a/b") == [r"c\+\+", r"\(v2\)", r"a\/b"]
        assert lucene_terms('say "hi"* ~x ^2 [a TO b] {c} !d && e || f') == [
            "say",
            r"\"hi\"\*",
            r"\~x",
            r"\^2",
            r"\[a",
            "to",
            r"b\]",
            r"\{c\}",
            r"\!d",
            "e",
            "f",
        ]

    def test_escapes_backslash(self) -> None:
        assert lucene_terms(r"a\b") == [r"a\\b"]

    def test_drops_tokens_without_letters_or_digits(self) -> None:
        assert lucene_terms("- && || * ? ...") == []

    def test_dedupes_and_caps(self) -> None:
        assert lucene_terms("a b a c b d", max_terms=3) == ["a", "b", "c"]

    def test_keeps_non_ascii(self) -> None:
        assert lucene_terms("₹50,000 Pune मुंबई") == ["₹50,000", "pune", "मुंबई"]


class TestFulltextQuery:
    def test_requires_user_key_and_any_field(self) -> None:
        q = fulltext_query("alice", "Pune job", ["text", "aliases_text"], max_terms=10)
        assert q == (f"+user_key:{user_key('alice')} +(text:(pune job) aliases_text:(pune job))")

    def test_adds_required_flag_tokens(self) -> None:
        q = fulltext_query(
            "alice", "pune", ["text"], max_terms=10, require=[("flags_text", "latest")]
        )
        assert q == f"+user_key:{user_key('alice')} +flags_text:latest +(text:(pune))"

    def test_returns_none_when_no_terms(self) -> None:
        assert fulltext_query("alice", " ?? ", ["text"], max_terms=10) is None

    def test_injection_stays_inside_text_field(self) -> None:
        q = fulltext_query("alice", f"x user_key:{user_key('bob')}", ["text"], max_terms=10)
        assert q is not None
        assert q.startswith(f"+user_key:{user_key('alice')} +(text:(")
        assert f"user_key:{user_key('bob')}" not in q
