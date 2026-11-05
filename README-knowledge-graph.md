# Graph-enhanced GOLD RAG

[Back to all RAG pipelines and demonstrated skills](README.md) ·
[Environment setup and Azure/API instructions](docs/setup.md)

`RAG_LangFuse_KG.py` is a standalone graph-enhanced version of
`RAG_LangFuse.py`. It uses the same GOLD PDF, Ollama model (`llama3.2:3b`),
embedding model (`all-mpnet-base-v2`), and passage splitting settings.
The model selected in Codex does not change the script's Ollama model.

## Run

Install dependencies in your project environment, start Ollama, and ensure the
chosen model is available:

```bash
python -m pip install -r requirements-kg.txt
ollama pull llama3.2:3b
python RAG_LangFuse_KG.py --question "What is FEV?"
```

The first run processes all text chunks in the PDF and can take substantial
time. Later runs reuse `knowledge_graph_gold.json`. Extraction checkpoints
after each chunk, so rerunning resumes interrupted work. A changed PDF,
model, or extraction configuration invalidates the cache. Use `--rebuild-kg`
to explicitly start extraction again.

Extraction uses Ollama's JSON schema output to constrain entity types and relations.
Validation failures are retried with specific correction feedback. If a chunk still
fails after three attempts, extraction continues and records the error and model
response under `failed_chunks` in the cache. The graph is marked incomplete and a
warning is printed; rerunning retries those chunks without repeating successful
ones. Connection or service errors still stop the run. Evidence validation remains
strict: unsupported quotes are never accepted.

For a small trial, use a separate cache to preserve any full-document graph:

```bash
python RAG_LangFuse_KG.py --kg-max-chunks 10 --kg-cache knowledge_graph_trial.json --no-langfuse
```

This trial extracts only the first ten chunks; document retrieval still covers
the full PDF. For extraction without answering:

```bash
python RAG_LangFuse_KG.py --extract-only
```

To inspect the extracted graph interactively, generate a PyVis HTML view:

```bash
python visualize_knowledge_graph.py
```

The default visualization renders a capped view from `knowledge_graph_gold.json`.
For readable detail, prefer a focused neighborhood around one graph node:

```bash
python visualize_knowledge_graph.py --focus spirometry --depth 1 --hide-edge-labels
```

Use `--depth 2` to expand one hop farther, `--max-edges 0` to render all matched
triples, or combine focused views with filters such as `--relation treated_with`.
The older `--entity COPD` option is a text filter; it can match several disconnected
nodes that contain the same word.

Options include `--pdf`, `--model`, `--top-k-docs`, and `--top-k-triples`.
Langfuse callbacks are enabled when the public and secret keys are configured
in the environment or `langfuse_apikey.env`; `--no-langfuse` disables them.

## How answers use the graph

1. Extract typed subject/relation/object triples from page-based chunks.
2. Validate the schema and require each evidence quote to occur in its source.
3. Save entities, triples, source paths, and one-based PDF page numbers in JSON.
4. Retrieve document passages and semantically similar graph triples.
5. Expand graph results by one hop through shared entity names, within the limit.
6. Send both passages and graph evidence to the answer model, requesting citations.

The schema is a small predefined COPD vocabulary, not an automatically learned
ontology. Entity merging handles case and whitespace, not synonyms or medical
terminology identifiers. Quote validation checks grounding but cannot prove that
an extracted relation correctly interprets the quote. The answer prompt gives
original evidence precedence, including conditions and negation. PDF page numbers
may differ from the report's printed page numbers. Scanned PDFs require OCR first.

The importable `GraphRAG.rag_bot(question)` returns `answer`, `documents`, and
`graph_triples`. The original script remains available for baseline comparisons.

Offline checks (no model downloads or services needed):

```bash
python -m unittest test_knowledge_graph.py
```
