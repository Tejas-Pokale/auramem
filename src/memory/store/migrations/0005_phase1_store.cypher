// Schema the Phase 1 store relies on.

// Exactly one ObservationLog per user (save uses MERGE plus an optimistic version check).
// The uniqueness constraint's backing index replaces the plain range index from 0004.
DROP INDEX observation_log_user_id IF EXISTS;
CREATE CONSTRAINT observation_log_user_id_unique IF NOT EXISTS
FOR (o:ObservationLog) REQUIRE o.user_id IS UNIQUE;

// seq is unique within a session. The backing index also serves NEXT linking and the
// neighbor-turn lookups used for context expansion.
CREATE CONSTRAINT turn_session_seq IF NOT EXISTS
FOR (t:Turn) REQUIRE (t.session_id, t.seq) IS UNIQUE;

// Full-text indexes cannot filter on booleans, so facts also carry flags_text, a
// space-separated pair of tokens: "latest"|"superseded" and "factual"|"hypothetical".
// BM25 queries require these tokens so the is_latest / is_hypothetical filters run inside
// the index, like the vector index's WITH [...] filters. The store writes both forms.
// No Fact nodes existed before this migration, so no backfill is needed.
DROP INDEX fact_text IF EXISTS;
CREATE FULLTEXT INDEX fact_text IF NOT EXISTS
FOR (f:Fact) ON EACH [f.text, f.aliases_text, f.user_key, f.flags_text];
