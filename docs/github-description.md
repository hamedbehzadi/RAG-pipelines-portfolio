# GitHub and portfolio descriptions

## GitHub About description

Copy this into the repository's **About → Description** field:

```text
Python RAG portfolio: local vector search, Langfuse tracing, LangGraph workflows, knowledge graph augmentation, and Azure hybrid search over GOLD COPD guidelines, with a FastAPI service and Docker packaging.
```

## Repository

[`hamedbehzadi/RAG-pipelines-portfolio`](https://github.com/hamedbehzadi/RAG-pipelines-portfolio)

## Suggested GitHub topics

```text
rag, retrieval-augmented-generation, python, langchain, langgraph,
langfuse, ollama, huggingface, knowledge-graph, graph-rag,
azure-ai-search, hybrid-search, fastapi, docker, llm, portfolio
```

## Project description

I implemented a collection of retrieval-augmented generation pipelines using the GOLD 2025 COPD report as a shared document corpus. Starting with local PDF ingestion, sentence-transformer embeddings, and vector retrieval, I extended the project with Langfuse observability, LangGraph workflow orchestration, and a knowledge graph pipeline that combines document passages with validated, source-linked triples. I also implemented Azure AI Search ingestion and hybrid retrieval, connected a hosted chat model, and packaged the Azure question-answering workflow as a FastAPI service with Docker.

This project demonstrates practical experience with document preparation, semantic and hybrid search, context-grounded prompting, structured LLM output, graph evidence retrieval, tracing, backend APIs, and container packaging. Evaluation helpers and offline graph tests support experimentation; no clinical validation or comparative performance benchmark is claimed.

## CV or LinkedIn bullets

- Implemented five RAG variants covering local vector retrieval, Langfuse tracing, LangGraph orchestration, knowledge graph augmentation, and Azure hybrid search.
- Built a resumable knowledge graph extraction workflow with structured JSON output, schema and evidence validation, retry feedback, source provenance, and bounded one-hop expansion.
- Developed a FastAPI question-answering service returning cited sources and PDF page references, with Docker packaging and environment-based Azure configuration.
- Added lightweight retrieval, keyword-coverage, and abstention evaluation helpers, alongside mocked tests for graph extraction and retrieval behavior.

## Suggested walkthrough for reviewers

1. Start with the pipeline comparison and skills table in the main README.
2. Read `main.py` for the baseline document-to-answer flow.
3. Compare `RAG_LangFuse.py` and `RAG_LangGraph.py` for tracing and explicit workflow stages.
4. Inspect `RAG_LangFuse_KG.py` and its tests for structured evidence and extraction reliability.
5. Follow Azure ingestion, querying, and `app.py` to see the same use case exposed through a cloud-backed API.
6. Run the offline tests and generate a graph visualization with a local document/cache.

These texts are prepared for use on GitHub and in a portfolio. They do not change the remote repository's About field or publish the local files.
