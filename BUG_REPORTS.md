---
  Bug Report: Code indexing entity/relation counts don't match Cypher query results

  Environment

  - simplemem MCP server (PROD mode)
  - Project indexed: /weka/home-shimon/repo/3dtex

  Steps Performed

  1. Indexed directory:
  index_directory(
    path="/weka/home-shimon/repo/3dtex",
    patterns=["**/*.py", "**/*.yaml", "**/*.yml"],
    clear_existing=true
  )
  Result: {"files_indexed": 420, "chunks_created": 5507}

  2. Queried stats:
  get_stats()
  Result:
  {
    "total_memories": 0,
    "total_relations": 10554,
    "types_breakdown": {
      "function": 823,
      "class": 766,
      "module": 430,
      "file": 414
    },
    "entities": 2433,
    "code_chunks": 6366
  }

  3. Tested semantic search:
  search_code(
    query="VAE encoder forward pass",
    project_root="/weka/home-shimon/repo/3dtex",
    limit=5
  )
  Result: Returns 5 relevant code chunks with scores 0.28-0.43. ✅ Working.

  4. Tested Cypher queries:

  | Query              | Params                    | Result                             |
  |--------------------|---------------------------|------------------------------------|
  | count_by_type      | {"type": "function"}      | {"count": 0}                       |
  | count_by_type      | {"type": "class"}         | {"count": 0}                       |
  | count_by_type      | {"type": "file"}          | {"count": 0}                       |
  | count_by_type      | {"type": "module"}        | {"count": 0}                       |
  | count_memories     | {}                        | {"count": 0}                       |
  | recent_memories    | {"limit": 10}             | {"results": []}                    |
  | get_entities       | {"limit": 20}             | {"error": "Internal server error"} |
  | get_entity_by_name | {"name": "ColorFieldVAE"} | {"error": "Internal server error"} |

  5. Queried code_stats:
  code_stats(project_root="/weka/home-shimon/repo/3dtex")
  Result: {"chunk_count": 6366, "unique_files": 0}

  Observed Inconsistencies

  | Metric    | get_stats() | Cypher Query             |
  |-----------|-------------|--------------------------|
  | Entities  | 2,433       | 0                        |
  | Relations | 10,554      | N/A (no nodes to relate) |
  | Functions | 823         | 0                        |
  | Classes   | 766         | 0                        |
  | Modules   | 430         | 0                        |
  | Files     | 414         | 0                        |
  | Memories  | 0           | 0                        |

  | Metric         | index_directory result | code_stats()       |
  |----------------|------------------------|--------------------|
  | Files indexed  | 420                    | unique_files: 0    |
  | Chunks created | 5,507                  | chunk_count: 6,366 |

  Graph Schema Reference

  From get_graph_schema(), these node types are defined:
  - Entity with properties: name, type, version, created_at, last_modified
  - CodeChunk with properties: uuid, filepath, project_root, start_line, end_line, created_at
  - Relationship REFERENCES from Memory|CodeChunk to Entity

  Facts

  1. search_code returns results → vector index contains data
  2. get_stats shows 2,433 entities and 10,554 relations → some storage has this metadata
  3. All Cypher queries return 0 or error → graph database has no nodes
  4. code_stats returns unique_files: 0 after indexing 420 files
  5. No process_trace or store_memory was called - only index_directory

  ---
