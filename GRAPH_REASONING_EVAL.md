# Graph Reasoning Evaluation

**Evaluator:** Gemini 3 Pro Preview
**Date:** 2024-12-30
**Confidence:** 9/10

## Verdict

The implementation provides a solid **Context-Augmented Retrieval** system rather than true "Graph Reasoning." It effectively traverses hierarchical and temporal links (Session hierarchy) but lacks the semantic connectivity (Shared Entity nodes) required to reason across disjoint sessions.

## Current State

- **Graph:** KuzuDB with 259 nodes, 722 edges (2.8 edges/node)
- **Node Types:** session_summary, chunk_summary, message
- **Edge Types:** contains, child_of, follows, supports, relates
- **Scoring:** `seed_similarity × Π(edge_weights) × temporal_decay`

## Strengths

| Aspect | Assessment |
|--------|------------|
| Hierarchical Design | Session→Chunk→Message mirrors LLM interactions perfectly |
| Bi-directional Traversal | `child_of` + `direction="both"` enables zoom in/out |
| Temporal Decay | ~80 day half-life well-tuned for engineering memory |
| Conservative Propagation | Multiplicative weights prevent result drift |

## Critical Weaknesses

### 1. Missing Knowledge Layer

While `traces.py` extracts entities (files, commands, tools), it stores them as *metadata* on the Message node rather than creating distinct Entity nodes in the graph.

- **Current:** `Message(metadata: "file: main.py")`
- **Needed:** `Message --[references]--> Node(File: "main.py") <--[references]-- Message`

**Consequence:** Cannot traverse from "Session A" to "Session B" via a shared file or tool. The graph is effectively a set of disconnected trees (one per session).

### 2. Low Connectivity

With 2.8 edges/node (mostly structural parent/child/prev/next), the graph is sparse. True reasoning usually requires denser semantic connections.

### 3. Double Penalty on Path Length

A 3-hop path via `follows` edges: `0.6³ = 0.216` crushes the score. If the vector search seed score is 0.75, the 3-hop neighbor gets 0.16, likely filtered out.

### 4. Semantic Blindness

Edge weights are static. A `follows` link is treated the same whether messages are milliseconds or hours apart.

### 5. Super-Node Explosion Risk

If a SessionSummary is selected as a seed, it connects to every ChunkSummary with weight 1.0. A single generic match could flood results with every chunk from that session.

## Recommended Improvements

### 1. Promote Entities to Nodes

Modify `traces.py` to create shared nodes for FilePaths, ToolNames, and ErrorTypes.

```
Message A --(references)--> Node(File: "main.py") <--(references)-- Message B
```

**Benefit:** Enables "How has `main.py` changed?" queries to gather context across all sessions.

### 2. Dynamic Edge Weights

Adjust `follows` weights based on time difference:
- Messages within 1 minute: 0.9
- Messages within 1 hour: 0.7
- Messages > 1 hour apart: 0.5

### 3. Cross-Session Reasoning

Entity nodes become bridges between sessions, enabling true multi-hop reasoning across the entire knowledge base.

## Key Takeaways

1. **Graph is Disconnected:** Current graph is collection of isolated Session trees
2. **Effective Context Retrieval:** Excels at "retrieving context around a match"
3. **Entity Promotion Required:** Move extracted entities from JSON metadata to first-class Graph Nodes
4. **Scoring Tuned for Local Context:** Heavy penalties on `follows`/`relates` prioritize hierarchical context

## Implementation Roadmap

- [ ] Migrate to FalkorDB for better graph capabilities
- [ ] Create Entity node types (FilePath, ToolName, ErrorType, Concept)
- [ ] Add `references` edges from Messages to Entities
- [ ] Implement dynamic edge weights based on temporal proximity
- [ ] Update reason() to leverage cross-session entity links
