// Range indexes (PLAN.md section 4).

CREATE INDEX fact_user_id IF NOT EXISTS FOR (f:Fact) ON (f.user_id);
CREATE INDEX fact_start IF NOT EXISTS FOR (f:Fact) ON (f.start);
CREATE INDEX fact_end IF NOT EXISTS FOR (f:Fact) ON (f.end);
CREATE INDEX fact_mentioned_at IF NOT EXISTS FOR (f:Fact) ON (f.mentioned_at);
CREATE INDEX fact_attribute_key IF NOT EXISTS FOR (f:Fact) ON (f.attribute_key);
CREATE INDEX turn_created_at IF NOT EXISTS FOR (t:Turn) ON (t.created_at);

// Not listed in PLAN.md section 4 but needed by every per-user lookup outside the
// vector/full-text indexes (sessions list, turn windows, entity scans, the log).
CREATE INDEX turn_user_id IF NOT EXISTS FOR (t:Turn) ON (t.user_id);
CREATE INDEX session_user_id IF NOT EXISTS FOR (s:Session) ON (s.user_id);
CREATE INDEX entity_user_id IF NOT EXISTS FOR (e:Entity) ON (e.user_id);
CREATE INDEX observation_log_user_id IF NOT EXISTS FOR (o:ObservationLog) ON (o.user_id);
