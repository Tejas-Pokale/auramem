// Let turn vector search filter by created_at inside the index, for the planner's time range.
// Existing turns are re-indexed from their stored embeddings; nothing is re-embedded.

DROP INDEX turn_vec IF EXISTS;
CREATE VECTOR INDEX turn_vec IF NOT EXISTS
FOR (t:Turn) ON t.embedding
WITH [t.user_id, t.created_at]
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'none'
}};
