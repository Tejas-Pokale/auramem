// Full-text (Lucene BM25) indexes (PLAN.md section 4).
//
// SEARCH ... WHERE filtering only exists for vector indexes; full-text search takes a
// Lucene query string. To filter user_id inside the index, every node also carries
// user_key = sha256(user_id) as lowercase hex. The standard analyzer keeps that as one
// token, so queries require it with a clause like  +user_key:<hex>  next to the text terms.
//
// aliases_text is the space-joined form of the aliases list.

CREATE FULLTEXT INDEX turn_text IF NOT EXISTS
FOR (t:Turn) ON EACH [t.text, t.user_key];

CREATE FULLTEXT INDEX fact_text IF NOT EXISTS
FOR (f:Fact) ON EACH [f.text, f.aliases_text, f.user_key];

CREATE FULLTEXT INDEX entity_text IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases_text, e.user_key];
