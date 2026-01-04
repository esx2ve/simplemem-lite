 ---
  Bug Report: Processed session memories exist in graph but are not searchable/queryable

  Environment

  - simplemem MCP server (PROD mode)
  - Project: /weka/home-shimon/repo/3dtex

  Steps Performed

  1. Processed two session traces:
  process_trace(session_id="01db03c9-ed8a-4395-8482-9e1be68e31f8")
  # Result: {"session_summary_id": "f7534919-99f6-4ec6-a4ee-706480169aa3", "chunk_count": 9, "message_count": 4}

  process_trace(session_id="372b8ec5-fd6d-4297-ab6e-a58b82d1d059")
  # Result: {"session_summary_id": "decb058c-b15f-412c-b653-612d728f24ca", "chunk_count": 2, "message_count": 1}

  2. Verified memories exist via stats:
  get_stats()
  # Returns: {"total_memories": 19, ...}

  3. Verified memories exist via Cypher count:
  run_cypher_query(query="count_memories", params={})
  # Returns: {"results": [{"count": 19}]}

  run_cypher_query(query="count_by_type", params={"type": "session_summary"})
  # Returns: {"results": [{"count": 2}]}

  run_cypher_query(query="count_by_type", params={"type": "chunk_summary"})
  # Returns: {"results": [{"count": 11}]}

  run_cypher_query(query="count_by_type", params={"type": "message"})
  # Returns: {"results": [{"count": 5}]}

  4. Attempted semantic search - returns empty:
  search_memories(query="simplemem indexing session", limit=10)
  # Returns: {"results": []}

  search_memories(query="What was done in this session?", limit=10)
  # Returns: {"results": []}

  5. Attempted ask_memories - returns empty:
  ask_memories(query="What was done in this session?", project_id="/weka/home-shimon/repo/3dtex")
  # Returns: {"answer": "I don't have any relevant memories...", "memories_used": 0}

  ask_memories(query="What was done in this session?")  # without project_id
  # Returns: {"answer": "I don't have any relevant memories...", "memories_used": 0}

  6. Attempted reason_memories - returns empty:
  reason_memories(query="What issues were discovered with simplemem?", project_id="/weka/home-shimon/repo/3dtex")
  # Returns: {"conclusions": [], "count": 0}

  7. Attempted to retrieve memory content via Cypher - internal errors:
  run_cypher_query(query="recent_memories", params={"limit": 5})
  # Returns: {"error": "Internal server error"}

  run_cypher_query(query="get_memory_by_type", params={"type": "session_summary", "limit": 2})
  # Returns: {"error": "Internal server error"}

  run_cypher_query(query="get_memory", params={"uuid": "f7534919-99f6-4ec6-a4ee-706480169aa3"})
  # Returns: {"error": "Internal server error"}

  Summary Table

  | Operation                       | Expected            | Actual                 |
  |---------------------------------|---------------------|------------------------|
  | count_memories                  | 19                  | 19 ✅                  |
  | count_by_type (session_summary) | 2                   | 2 ✅                   |
  | count_by_type (chunk_summary)   | 11                  | 11 ✅                  |
  | count_by_type (message)         | 5                   | 5 ✅                   |
  | search_memories                 | Results             | Empty array            |
  | ask_memories                    | Answer with sources | "no relevant memories" |
  | reason_memories                 | Conclusions         | Empty array            |
  | recent_memories                 | Memory list         | Internal server error  |
  | get_memory                      | Memory content      | Internal server error  |
  | get_memory_by_type              | Memory list         | Internal server error  |

  Observed Behavior

  1. process_trace reports success with chunk/message counts
  2. get_stats and count_* Cypher queries confirm 19 Memory nodes exist
  3. Semantic search (search_memories) returns no results
  4. LLM-based queries (ask_memories, reason_memories) find no memories
  5. Cypher queries that return memory content fail with "Internal server error"
  6. Cypher queries that return counts succeed

  Facts

  - 19 Memory nodes exist in FalkorDB graph (confirmed by count queries)
  - Memory nodes have types: session_summary(2), chunk_summary(11), message(5), unknown(1)
  - Vector search returns empty for any query
  - Any Cypher template that would return memory content/properties fails
  - Only aggregate queries (counts) succeed

  ---

