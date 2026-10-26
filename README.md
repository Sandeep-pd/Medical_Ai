# Medical AI RAG

A FastAPI application that accepts healthcare documents, extracts and indexes
their text, retrieves relevant passages for a question, and sends those
passages to Gemini to draft a cited answer.

> **Not for clinical use:** generated answers can be wrong. Verify important
> statements against the source document and a qualified professional. The
> separate prediction endpoint is a synthetic-data demo, not a medical model.

## What it does

1. **Accepts a document:** CSV, DOCX, PDF, PNG, JPG, or JPEG.
2. **Extracts text:** reads CSV rows, DOCX paragraphs and tables, PDF text (with
   OCR for scanned pages), or image text using OCR.
3. **Chunks and indexes the text:** saves chunks in persistent ChromaDB with
   embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieves evidence for a question:** combines ChromaDB semantic matches
   with BM25 keyword matches using reciprocal-rank fusion.
5. **Builds cited sources:** includes document ID, page, chunk ID, and passage.
6. **Generates an answer:** asks Gemini to answer from the evidence, preserve
   values and units, cite source labels, and not infer a patient diagnosis.

Gemini models are tried in configured order. The default order is
`models/gemini-3.5-flash-lite`, `models/gemini-3.5-flash`, then
`models/gemini-3.6-flash`. Each attempt has a 10-second timeout. A successful
model stops the fallback sequence; if all fail, the API returns a structured
error. If all three models time out, a request can take up to about 30 seconds.

## Project structure

```text
Medical_AI/
├── app/
│   ├── main.py                  # FastAPI app, health endpoint, CLI
│   ├── config.py                # Environment settings and storage paths
│   ├── api/
│   │   ├── documents.py         # Upload and ingest documents
│   │   ├── questions.py         # Retrieve evidence and answer questions
│   │   └── predictions.py       # Synthetic demo prediction endpoints
│   ├── ingestion/
│   │   ├── layout_parser.py     # CSV, DOCX, and PDF extraction
│   │   ├── ocr.py               # Image/scanned-page OCR
│   │   └── lab_extractor.py     # Extract matching laboratory result patterns
│   ├── rag/
│   │   ├── chunking.py          # Split extracted text into chunks
│   │   ├── embeddings.py        # Load the sentence-transformer
│   │   ├── indexing.py          # Persistent ChromaDB write and search
│   │   ├── hybrid_search.py     # BM25 keyword search and rank fusion
│   │   └── citations.py         # Format evidence and source labels
│   ├── llm/
│   │   ├── generator.py         # Answer-generation entry point
│   │   ├── crew.py              # Gemini calls, model fallback, error mapping
│   │   └── safety.py            # Question and citation validation helpers
│   ├── ml/                      # Separate synthetic demo prediction model
│   └── schemas/                 # Validated API request/response models
├── data/
│   ├── medical.csv              # Default CSV for the summary CLI
│   ├── documents/               # Uploaded source files
│   └── chroma/                  # Persistent ChromaDB files
├── models/                      # Reserved model storage
├── tests/test_core.py           # Focused pipeline and API tests
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Technologies used

| Technology | Purpose |
|---|---|
| Python | Application language |
| FastAPI, Uvicorn | HTTP API and development server |
| Pydantic | Validate API requests and responses |
| ChromaDB | Persistent vector database for document chunks |
| Sentence Transformers | Create text embeddings for semantic retrieval |
| BM25 (`rank-bm25`) | Keyword retrieval, combined with vector search |
| Google Gen AI (`google-genai`) | Generate evidence-grounded answers with Gemini |
| PyMuPDF, python-docx, pandas | Read PDF, DOCX, and CSV documents |
| PaddleOCR, Pillow | OCR for scanned documents and images |
| scikit-learn, XGBoost | Supporting dependencies for the demo ML pipeline |

## Requirements and installation

Use Python 3.10 or newer. From PowerShell at the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` or `GOOGLE_API_KEY` to a key with Gemini
API access. Keep the key private; `.env` is excluded from Git. Do not put API
keys in source code or paste them into logs or screenshots.

OCR and embedding packages can be large. On the first upload or question, the
embedding model is loaded and may be downloaded. Later questions reuse it.
Gemini API calls require internet access.

## Configuration

Settings are loaded in [`app/config.py`](app/config.py). Shell environment
variables override values from the project-root `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Base data directory |
| `DOCUMENTS_DIR` | `data/documents` | Uploaded source files |
| `CHROMA_DIR` | `data/chroma` | Persistent ChromaDB data |
| `MODEL_DIR` | `models` | Reserved model directory |
| `DATASET_PATH` | `data/medical.csv` | CSV used by the summary CLI |
| `MAX_UPLOAD_BYTES` | `20971520` | Maximum upload size (20 MiB) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Gemini API key; set at least one |
| `GEMINI_MODEL_IDS` | See below | Comma-separated model fallback order |

Default model order:

```text
models/gemini-3.5-flash-lite,models/gemini-3.5-flash,models/gemini-3.6-flash
```

Change the fallback sequence by editing `GEMINI_MODEL_IDS` in `.env`. Retrieval
defaults are set in code: 800-character chunks, 120-character overlap, up to 10
search candidates, and up to 3 cited sources.

## Run the API

Start from the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If port 8000 is already in use, use another port, for example `--port 8001`.
The server stays in the terminal until stopped with **Ctrl+C**.

Useful URLs (replace the port if necessary):

| URL | What to check |
|---|---|
| `http://127.0.0.1:8000/health` | API health; should return `{"status":"ok"}` |
| `http://127.0.0.1:8000/docs` | Swagger UI to try the API interactively |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI definition |

## Try uploading and asking a question

### 1. Check health

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected response:

```text
status
------
ok
```

### 2. Upload a document

Use a real local file path:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents" `
  -F "file=@C:\path\to\report.pdf"
```

Supported extensions are `.csv`, `.docx`, `.pdf`, `.png`, `.jpg`, and `.jpeg`.
The response reports the generated `document_id`, page count, chunk count, and
any extracted laboratory results. The original file is saved under
`data/documents/`; its chunks are stored in `data/chroma/`.

### 3. Ask a question

PowerShell example:

```powershell
$body = @{ question = "What hemoglobin value and reference range are reported?" } |
  ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/questions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

The response contains the question, generated answer, and cited sources. In
Swagger UI, expand **POST `/api/questions`**, select **Try it out**, replace the
example with valid JSON, and click **Execute**:

```json
{"question":"What hemoglobin value and reference range are reported?"}
```

Use `ConvertTo-Json` in PowerShell or a JSON-aware client to avoid malformed
request bodies. `422 Unprocessable Content` commonly means the request body
wasn't valid JSON or the `question` field was empty or longer than 2000
characters.

### 4. Check whether a question succeeded

- **200:** answer and cited sources returned.
- **404:** upload a document before asking.
- **422:** check the request JSON and question length.
- **429:** all configured models reported quota exhaustion.
- **503:** all configured models failed or Gemini is temporarily unavailable.
  Check the server terminal, API key, model access, and internet connection.

The first question can take longer because it loads the embedding model. Warm
retrieval reuses the model and ChromaDB collection. The model fallback affects
answer generation, not vector retrieval.

## Run the tests

From the project root:

```powershell
python -m unittest discover -s tests -p test_core.py
```

The focused suite checks ingestion, chunking, ChromaDB integration, retrieval
fusion, the question API response, and Gemini fallback/error handling. Tests
mock external Gemini requests; a passing suite does not verify the live API key
or remote model availability.

## Other available commands and endpoints

### Synthetic prediction demo

`POST /api/predictions` accepts exactly four numeric features. This is a
synthetic demonstration only; it is unrelated to uploaded documents and must
not be used for patient diagnosis or treatment.

```powershell
$body = @{ features = @(0.8, -0.3, 0.5, 0.1) } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/predictions" `
  -Method Post -ContentType "application/json" -Body $body
```

`GET /api/predictions/evaluation` reports metrics on synthetic demo data.

### CSV summary CLI

Print a summary of the configured dataset:

```powershell
python -m app.main --summary-only
```

Choose a different CSV:

```powershell
python -m app.main --summary-only --dataset "C:\path\to\data.csv"
```

### Docker

Build and run the API:

```powershell
docker build -t medical-ai .
docker run --rm -p 8000:8000 --env-file .env `
  -v "${PWD}\data\chroma:/app/data/chroma" `
  medical-ai
```

The volume keeps the ChromaDB index when the container is removed.

## Important limitations

- This is a document search and answer application, not a clinical decision
  system. Verify outputs against the uploaded source and a qualified clinician.
- Gemini can be unavailable, rate limited, or return incorrect answers.
- Uploaded documents and their extracted text may contain sensitive health
  information. Store and process them only in an environment authorized for
  that data.
- The prediction demo uses synthetic data and does not analyze the uploaded
  healthcare CSV.
- Uploaded files are retained in `data/documents/`; there is no delete-document
  endpoint yet.
