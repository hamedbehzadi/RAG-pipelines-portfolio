# Setup and execution

[Back to the project overview](../README.md)

Run all commands from the repository root with Python 3.11. The commands below use Bash.

## 1. Create an environment and choose dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the Azure API only:

```bash
python -m pip install -r requirements.txt
```

For local pipelines, graph extraction/visualization, Azure ingestion, and LangGraph:

```bash
python -m pip install -r requirements-kg.txt langgraph
```

`requirements-kg.txt` includes `requirements.txt` and adds local RAG/graph dependencies, including `pypdf`. `langgraph` is installed explicitly because neither existing requirements file lists it. Dependencies are not fully pinned to a tested environment.

## 2. Provide the document

Obtain a copy of the GOLD report you are permitted to use and place it at:

```text
./GOLD-2025-Report-v1.0-15Nov2024_WMV.pdf
```

PDFs are excluded from Git. The scripts retain physical PDF page positions, displayed as one-based page numbers; these may differ from printed page numbers. Scanned documents require OCR outside this project.

## 3. Run local RAG with Ollama

Install Ollama separately and ensure its service is running. If it is not running as a background service, start `ollama serve` in another terminal.

```bash
ollama pull llama3.2:3b
python main.py
```

The embedding model is downloaded on first use, and the baseline rebuilds its in-memory index each time. Edit the final `question` assignment in `main.py` to change its example question.

## 4. Enable Langfuse tracing

```bash
cp langfuse_apikey.env.example langfuse_apikey.env
```

Fill in your actual Langfuse public key, secret key, and instance base URL in the copied file. The example URL must be replaced with your instance's URL.

```bash
python RAG_LangFuse.py
python RAG_LangGraph.py
```

Both scripts initialize Langfuse and call its authentication check. The graph-enhanced implementation enables callbacks when credentials are present and also supports `--no-langfuse`.

## 5. Run graph-enhanced RAG

Start with a small extraction run and a separate cache:

```bash
python RAG_LangFuse_KG.py \
  --question "What is FEV?" \
  --kg-max-chunks 10 \
  --kg-cache knowledge_graph_trial.json \
  --no-langfuse
```

Only graph extraction is limited to ten chunks; passage retrieval still covers the full PDF. For the full graph:

```bash
python RAG_LangFuse_KG.py --question "What is FEV?" --no-langfuse
```

The full run writes `knowledge_graph_gold.json`. Successful chunks are reused when the cache signature matches, and failed validation chunks are retried on a later run. Changing the absolute source path, PDF contents, model, or extraction configuration can invalidate the cache. Use `--extract-only` to skip answering or `--rebuild-kg` to rebuild the selected cache.

To visualize the trial graph:

```bash
python visualize_knowledge_graph.py \
  --input knowledge_graph_trial.json \
  --entity '' \
  --max-edges 80 \
  --output knowledge_graph_trial.html
```

Open the generated HTML file in a browser. GitHub displays HTML source rather than running this visualization inline. The visualizer defaults to filtering for `spirometry`; `--entity ''` makes that text filter match all entities. A focused full-graph view can use:

```bash
python visualize_knowledge_graph.py --focus spirometry --depth 1 --hide-edge-labels
```

See the [graph guide](../README-knowledge-graph.md) for additional behavior and limitations.

## 6. Configure Azure and ingest the PDF

You need an Azure AI Search service, an index name, a search key permitted to create/write the index for ingestion, and an Azure-hosted chat deployment accessible through the OpenAI-compatible base URL expected by the code. Resource provisioning is outside this repository.

```bash
cp .env.example .env
```

Fill in all six Azure variables in `.env`. `AZURE_OPENAI_CHAT_DEPLOYMENT` must identify your deployment. Set `AZURE_OPENAI_BASE_URL` to the compatible API base URL for that deployment, including its required path; the source code passes it directly to `ChatOpenAI` rather than constructing an Azure URL.

```bash
python RAG_Azure_ingest
python test_Azure_chatmodel_connection.py
python RAG_Azure_quesry
```

The ingestion script writes documents to Azure and performs a retrieval check. Run it before querying a new index. There is no explicit stable-ID or deduplication workflow, so repeated ingestion may add duplicate chunks. Query and API execution reuse the configured index; they do not ingest the PDF.

## 7. Start the API

With the Azure index populated and `.env` configured:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/docs` to use the interactive API documentation, or send a request:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does COPD mean?"}'
```

`POST /ask` returns an `answer` and a `sources` list. Each source contains `citation`, `source`, and `pdf_page`. Startup initializes the embedding model and Azure clients. The health endpoint reports application status without checking downstream services.

## 8. Run the API in Docker

```bash
docker build -t gold-rag-api .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 gold-rag-api
```

The build downloads CPU-only PyTorch and the embedding model. It copies `app.py`, not the PDF or local scripts, and requires a previously populated Azure index. Credentials are supplied at runtime using `--env-file`.

## 9. Run offline checks

```bash
python -m unittest test_knowledge_graph.py
```

These tests use mocks and temporary caches. They do not require the full dependency installation or call external services. The Azure connectivity script is a separate live check and may incur model usage charges.

## Common issues

| Symptom | What to check |
| --- | --- |
| Missing PDF | Match the expected filename and run from the repository root; the graph CLI also accepts `--pdf`. |
| Ollama connection failure | Ensure its service is running and the selected model has been pulled. |
| Missing `langgraph` or PDF dependencies | Use the full installation command in section 1. |
| Missing Azure environment variable | Populate `.env` with the names in `.env.example`. |
| Azure model connection failure | Check deployment name, API key, and the complete compatible base URL. |
| Empty Azure retrieval | Confirm ingestion succeeded against the same index used for querying. |
| Incomplete graph | Inspect `failed_chunks` in the cache and rerun to retry failed validation chunks. |
| No visualization matches | Check entity/relation filters; the default entity filter is `spirometry`. |

## Files kept local

The repository ignore rules exclude `.env`, `*.env`, `keys.txt`, PDFs, generated graph JSON/HTML, Python caches, and local IDE settings. Commit the placeholder example files, not actual credentials. If any credential has previously been shared, revoke it through its provider.
