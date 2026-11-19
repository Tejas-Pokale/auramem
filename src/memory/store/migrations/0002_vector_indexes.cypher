// Vector indexes (PLAN.md section 4). Cosine, 768 dims (EmbeddingGemma).
//
// Properties listed in WITH [...] can be filtered inside the index with SEARCH ... WHERE,
// which only accepts equality/range predicates joined by AND. That is why is_latest and
// is_hypothetical are booleans that are always set, never null.
// is_hypothetical is not in the PLAN.md example DDL, but section 5 step 5 requires
// hypotheticals to be excluded from default retrieval, so it must be filterable in-index.
//
// Quantization is disabled: the 2026.08+ default is binary, which trades recall for memory.
// Per-user vector counts are small, so we keep full precision.

CREATE VECTOR INDEX turn_vec IF NOT EXISTS
FOR (t:Turn) ON t.embedding
WITH [t.user_id]
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'none'
}};

CREATE VECTOR INDEX fact_vec IF NOT EXISTS
FOR (f:Fact) ON f.embedding
WITH [f.user_id, f.is_latest, f.is_hypothetical]
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'none'
}};

CREATE VECTOR INDEX entity_vec IF NOT EXISTS
FOR (e:Entity) ON e.embedding
WITH [e.user_id]
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'none'
}};
