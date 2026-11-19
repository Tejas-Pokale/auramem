// Write path (Phase 4): the exact-duplicate guard for facts (PLAN.md section 5, step 8).
// dedup_key = sha256(user_id | subject entity | normalized text), computed by the store.
// A restatement of an existing fact merges into it (new mention date, max confidence,
// extra DERIVED_FROM edges) instead of creating a second node.

CREATE CONSTRAINT fact_dedup_key IF NOT EXISTS
FOR (f:Fact) REQUIRE f.dedup_key IS UNIQUE;
