// Uniqueness constraints (PLAN.md section 4). Each also creates a backing range index.

CREATE CONSTRAINT user_user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

CREATE CONSTRAINT session_session_id IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT turn_turn_id IF NOT EXISTS
FOR (t:Turn) REQUIRE t.turn_id IS UNIQUE;

CREATE CONSTRAINT fact_fact_id IF NOT EXISTS
FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE;

CREATE CONSTRAINT entity_entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
