# Graph Schema Analysis for SimpleMem-Lite

## Current Schema (v1 - Dec 2024)

### Nodes
| Type | Count | Subtypes |
|------|-------|----------|
| Memory | 69 | session_summary, chunk_summary, message |
| Entity | 26 | file (7), tool (10), command (6), error (3) |

### Relationships
| Type | Count | Purpose |
|------|-------|---------|
| RELATES_TO | 169 | Memory-to-memory (contains, child_of, supports, follows, similar) |
| REFERENCES | 56 | Memory-to-entity links |

---

## Multi-Model Consensus (8-9/10 confidence)

### Models Consulted
| Model | Stance | Confidence |
|-------|--------|------------|
| Gemini-2.5-Pro | FOR improvements | 9/10 |
| Gemini-2.5-Flash | AGAINST complexity | 8/10 |
| Gemini-3-Pro-Preview | NEUTRAL balanced | 9/10 |

### Unanimous Agreements
1. **Verb-specific edges** (READS, MODIFIES, EXECUTES, TRIGGERED) = highest priority, lowest complexity
2. **Goal/UserIntent nodes** = very high value for intent-based recall
3. **NEXT edges** = essential for temporal traversal
4. **Topic nodes** = DEFER - use embedding search on Goal nodes instead
5. **Entity Resolution** is THE critical success factor
6. **Extraction pipeline** is the real challenge, not schema complexity

### Key Insights
- "Lite" doesn't mean "dumb" - READS vs MODIFIES distinction is fundamental for code
- Goal nodes should be PRIMARY entry point for retrieval (the "why" → "what" chain)
- Current REFERENCES edge is too coarse for coding tasks
- Success depends on entity resolution - must canonicalize file paths

---

## Industry Research (Perplexity Deep Research)

### Key Frameworks Analyzed
- **Letta (MemGPT)**: Agent-managed memory blocks, hierarchical tiers
- **LangGraph**: State-based with reducers, Store interface for cross-thread memory
- **Zep (Graphiti)**: Three-tier subgraph (episodic → semantic → community), bi-temporal model
- **Mem0**: Flexible multi-backend, optional graph layer

### Common Patterns
1. **Memory Hierarchy**: In-context (pinned) vs External (retrieved)
2. **Dual Memory Types**: Episodic (events) vs Semantic (facts)
3. **Specific Verb Edges**: All frameworks use READS, MODIFIES, etc.
4. **Entity Resolution**: Critical for preventing memory fragmentation
5. **Temporal Validity**: Bi-temporal models track event time vs transaction time

---

## FINAL SCHEMA v2 (Grok-4 Synthesis)

### Node Types

#### Memory Node
```
Memory {
  uuid: String (primary key)
  content: String (summary text)
  type: Enum["session_summary", "chunk_summary", "message"]
  source: String (claude_trace, user, extracted)
  session_id: String
  created_at: Timestamp
}
```

#### Entity Node
```
Entity {
  id: String (canonicalized path - primary key)
  type: Enum["file", "tool", "command", "error"]
  name: String (human-readable)
  version: Integer (increment on modifications)
  last_updated: Timestamp
}
```

#### Goal Node (NEW)
```
Goal {
  id: String (UUID)
  intent: String (user's objective description)
  status: Enum["active", "completed", "abandoned"]
  session_id: String
  created_at: Timestamp
  embedding: Vector (for similarity search - replaces Topic nodes)
}
```

### Edge Types

#### Memory-to-Memory Edges
| Edge | Purpose | Properties |
|------|---------|------------|
| `CONTAINS` | Session → Chunk → Message hierarchy | - |
| `NEXT` | Temporal sequence (linked list) | sequence_id: Int |

#### Memory-to-Entity Edges (replaces REFERENCES)
| Edge | Purpose | Properties |
|------|---------|------------|
| `READS` | Memory read a file | timestamp |
| `MODIFIES` | Memory changed a file | timestamp, change_summary |
| `EXECUTES` | Memory ran a tool/command | timestamp, execution_context |
| `TRIGGERED` | Action caused an error | timestamp, trigger_condition |

#### Goal Edges
| Edge | Purpose | Properties |
|------|---------|------------|
| `HAS_GOAL` | Session → Goal | - |
| `ACHIEVES` | Memory → Goal (contributed to) | - |

#### Optional (Phase 3)
| Edge | Purpose | Properties |
|------|---------|------------|
| `CONTEXT_WINDOW` | Files in context but not referenced | window_start, window_end |

### Entity Resolution Strategy

#### Canonicalization Rules
```python
def canonicalize_entity_id(raw_path: str, entity_type: str) -> str:
    if entity_type == "file":
        # Normalize to absolute path, resolve symlinks
        return os.path.realpath(os.path.abspath(raw_path))
    elif entity_type == "tool":
        # Lowercase, strip prefixes
        return raw_path.lower().replace("mcp__", "").replace("__", ":")
    elif entity_type == "command":
        # Extract base command
        return raw_path.split()[0].lower()
    elif entity_type == "error":
        # Hash error type + message prefix
        return hashlib.sha256(f"{error_type}:{message[:100]}").hexdigest()[:16]
```

#### Resolution Process
1. **On Ingestion**: Canonicalize ID before node creation
2. **Lookup First**: Query for existing node with canonical ID
3. **Merge if Found**: Update properties (increment version if modified)
4. **Edge Attachment**: Always resolve to canonical IDs first

#### Fragmentation Prevention
- Periodic maintenance query to detect near-duplicates via embedding similarity
- Merge duplicates, preserve all edges on canonical node

---

## Phased Implementation Roadmap

### Phase 1: Verb-Specific Edges (Week 1)
**Goal**: Replace generic REFERENCES with precise action verbs

- [ ] Add new edge types: READS, MODIFIES, EXECUTES, TRIGGERED
- [ ] Add VERSION property to Entity nodes
- [ ] Implement canonicalization in `add_entity_node()`
- [ ] Update trace processor to detect action type from Claude logs
- [ ] Migrate existing REFERENCES edges (default to READS)
- [ ] Test: "Find all files modified in session X"

### Phase 2: Goal/Intent Nodes (Week 2)
**Goal**: Enable intent-based recall

- [ ] Add Goal node type to schema
- [ ] Add HAS_GOAL and ACHIEVES edges
- [ ] Extract user's initial prompt as Goal.intent
- [ ] Add embedding generation for Goal.intent
- [ ] Implement similarity search on Goal embeddings
- [ ] Test: "Find sessions where I tried to fix auth bugs"

### Phase 3: Temporal & Context (Week 3)
**Goal**: Enable execution replay and context awareness

- [ ] Add explicit NEXT edges between Memory nodes
- [ ] Add CONTEXT_WINDOW edge (optional)
- [ ] Build traversal queries for "chain of thought" replay
- [ ] Test: "Show me the sequence of actions in session X"

### Deferred (Future)
- Topic nodes (use Goal embeddings for now)
- Bi-temporal model (event time vs transaction time)
- Community detection / clustering
- Edge invalidation for contradictory facts

---

## Example Queries (Post-Implementation)

```cypher
-- Find all files I modified while working on auth
MATCH (g:Goal)-[:HAS_GOAL]-(s:Session)-[:CONTAINS]->(m:Memory)-[:MODIFIES]->(f:Entity)
WHERE g.intent CONTAINS 'auth'
RETURN DISTINCT f.name, f.version

-- Replay execution sequence for a session
MATCH (m:Memory {session_id: $sid})-[:NEXT*]->(next:Memory)
RETURN m.content, next.content ORDER BY m.created_at

-- Find sessions that touched the same file
MATCH (m1:Memory)-[:MODIFIES]->(f:Entity)<-[:MODIFIES]-(m2:Memory)
WHERE m1.session_id <> m2.session_id
RETURN m1.session_id, m2.session_id, f.name

-- Intent-based search (via vector similarity on Goal.embedding)
-- Handled in application layer, not Cypher
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Entity fragmentation rate | < 5% duplicates |
| Intent recall accuracy | > 80% relevant results |
| Query latency (1-hop) | < 50ms |
| Query latency (2-hop) | < 200ms |

---

## References

- [Zep Graphiti Paper](https://arxiv.org/html/2501.13956v1)
- [Letta Memory Docs](https://docs.letta.com/guides/agents/memory/)
- [LangGraph Memory](https://docs.langchain.com/oss/python/langgraph/memory)
- [Mem0 Graph Memory](https://docs.mem0.ai/platform/features/graph-memory)
