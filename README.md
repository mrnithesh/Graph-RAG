# Graph-RAG Recruitment Prototype (Energy Sector)

Prototype **Graph-RAG** system for an AI-powered recruitment platform in the energy sector.

It builds a **knowledge graph** (candidates, skills, certifications, domains, experience) and performs **graph-first retrieval** (seed entities → neighborhood expansion → evidence paths) instead of naive chunk-based RAG.

## What this does (high level)

- **Knowledge base**: a `networkx` knowledge graph with typed nodes/edges for:
  - **Candidate profiles**: skills, certifications, domains, employers, titles, summary
  - **Role requirements**: required skills/certs/domains (either from JSON or extracted from free text using an LLM)
- **Graph-RAG retrieval**:
  - **Seed**: from a role’s explicit requirements (or from a free-text query by embedding-to-entity similarity)
  - **Expand**: traverse the graph (configurable depth) to find connected candidates
  - **Score**: graph overlap + optional embedding similarity (OpenAI embeddings)
  - **Explain**: return **evidence paths** showing why each candidate matched

## Repo layout

- `data/`
  - `candidates.json`: sample candidate profiles (energy sector flavored)
  - `roles.json`: sample role requirements
- `graph_rag/`
  - `graph.py`: knowledge graph abstraction
  - `ingest.py`: load candidates/roles into the graph
  - `openai_client.py`: **official OpenAI Python SDK** wrapper (embeddings + chat)
  - `retriever.py`: Graph-RAG retrieval + scoring + evidence paths
  - `system.py`: main `GraphRAG` entry point
  - `cli.py`: runnable demo CLI

## Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key (required for embeddings and LLM extraction):

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Notes:
- Embeddings are cached at `.cache/graph_rag/embeddings.npz` to avoid recomputing.
- If `OPENAI_API_KEY` is not set, the prototype still works, but it falls back to **string match** for seeding and disables embedding scoring (less accurate).

## Run the demo

### 1) Match against a structured role (from `data/roles.json`)

```bash
python -m graph_rag.cli --role-id role_001
```

Or:

```bash
python -m graph_rag.cli --role-id role_002
```

### 2) Match from a free-text query (Graph-RAG seeding via embeddings)

```bash
python -m graph_rag.cli --query "Looking for OT security engineer with IEC 62443, SCADA, segmentation for pipelines"
```

### 3) Match from a free-text role (LLM extracts requirements → Graph-RAG)

```bash
python -m graph_rag.cli \
  --free-role-title "OT Cybersecurity Engineer (SCADA/ICS)" \
  --free-role-desc "Secure and harden SCADA environments for pipeline operations. Need IEC 62443 and segmentation. GICSP preferred."
```

### Useful flags

```bash
python -m graph_rag.cli --role-id role_001 --top 3 --depth 2
python -m graph_rag.cli --role-id role_001 --json
```

## How it works (Graph-RAG vs naive RAG)

### Naive RAG (what we avoid)

- Split candidate profiles into text chunks
- Embed chunks
- Retrieve top chunks by similarity
- Ask an LLM to answer using chunks

Problems for recruitment matching:
- Hard to enforce structured requirements (must-have certs, domains)
- Weak explainability (“why this person?”)
- Doesn’t leverage relationships (skill ↔ domain ↔ title ↔ company)

### Graph-RAG (this prototype)

1) **Build KG** from candidate and role data:
- candidate → HAS_SKILL → skill
- candidate → HAS_CERT → cert
- candidate → HAS_DOMAIN → domain
- candidate → WORKED_AT → company
- candidate → HAD_TITLE → title
- role → REQUIRES_SKILL/REQUIRES_CERT/REQUIRES_DOMAIN → entities

2) **Seed entities**
- From role requirements edges, or
- From free text by embedding similarity to entity nodes (OpenAI embeddings)

3) **Graph expansion**
- Traverse neighbors to find candidates connected to seeds (depth is configurable)

4) **Scoring**
- **Graph score**: how many seed entities the candidate can reach (plus direct matches)
- **Embedding score**: role text vs candidate profile text (optional, uses OpenAI embeddings)

5) **Evidence**
- For matched seed entities, compute shortest paths to show the “why” chain

## Next extensions (optional)

- Add a persistent graph DB (Neo4j) and store embeddings in a vector index
- Add synonym/ontology normalization for skills (e.g. “OT security” vs “ICS security”)
- Add hard constraints (must-have certs) and calibration for seniority/location
