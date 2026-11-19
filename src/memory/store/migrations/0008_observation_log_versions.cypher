// ObservationLog history (Phase 6): the Reflector snapshots the previous log before every
// rewrite, so a bad condensation can be rolled back.

CREATE CONSTRAINT observation_log_version_key IF NOT EXISTS
FOR (v:ObservationLogVersion) REQUIRE (v.user_id, v.version) IS UNIQUE;
