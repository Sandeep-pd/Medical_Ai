# Medical AI RAG

A small FastAPI application for uploading healthcare documents, extracting their
text, retrieving relevant passages, and asking questions about the uploaded
content. It also includes a CSV summary command and a demonstration prediction
endpoint.

> **Important:** The prediction endpoint uses a model trained on synthetic
> example data. It is not trained on the uploaded healthcare CSV and must not be
> used for diagnosis, treatment, or any clinical decision. Generated answers
> can also be wrong; verify important information against the source document
> and a qualified professional.

 1. Project layout

```text
medical-ai-rag/
├── app/
│   ├── main.py                 # FastAPI app and CLI
│   ├── config.py               # Paths and upload settings
│   ├── api/                    # Upload, question, and prediction routes
│   ├── ingestion/              # CSV, DOCX, PDF, and image text extraction
│   ├── rag/                    # Chunking, search, reranking, and citations
│   ├── ml/                     # Demo prediction and CSV summary
│   ├── llm/                    # Gemini CrewAI agents and safety checks
│   └── schemas/                # Request and response validation
├── data/
│   ├── medical.csv             # Default CSV for summary command
│   └── documents/              # Uploaded files
├── models/                     # Reserved model storage
├── tests/
├── requirements.txt
├── Dockerfile
└── .env.example
```

 2. Requirements and installation

Use Python 3.10 or newer. From the project root, create and activate a virtual
environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the local settings file from the sample:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to a key created in Google AI Studio.
Keep `.env` private; it is ignored by Git. Do not paste the key into source
files, screenshots, or support messages. If a key has been shared, revoke it
and use a new key.

If PowerShell blocks activation, run the commands using the virtual environment
interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements include OCR, document parsing, ML, and Gemini API dependencies.
Some packages are large and may take time to install. PaddleOCR may require
platform-specific installation adjustments; see its installation instructions
if installation fails. Gemini generation uses the Google GenAI API over the
internet and does not download a local text-generation model.

3. Configuration: where to set parameters

Application settings are read from environment variables in
[`app/config.py`](app/config.py). The file `.env.example` documents the
available names. The app loads a project-root `.env` file automatically when
`python-dotenv` is installed (it is included in `requirements.txt`). Environment
variables already defined in the shell take precedence over values from `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Base directory for data |
| `DOCUMENTS_DIR` | `data/documents` | Where uploaded files are saved |
| `MODEL_DIR` | `models` | Reserved model directory |
| `DATASET_PATH` | `data/medical.csv` | CSV used by the CLI summary |
| `MAX_UPLOAD_BYTES` | `20971520` (20 MiB) | Maximum upload size |
| `GEMINI_API_KEY` | unset | Required key for Gemini answer generation |
| `GEMINI_MODEL_ID` | `gemini-3.8-flash` | Gemini model used to generate answers |
| `CREWAI_TRACING_ENABLED` | `false` for this app's Crew | Keep CrewAI tracing prompts/sharing disabled |

Example PowerShell configuration:

```powershell
$env:DATASET_PATH = "C:\path\to\my-healthcare.csv"
$env:MAX_UPLOAD_BYTES = "52428800"
$env:GEMINI_API_KEY = "your-api-key"
$env:GEMINI_MODEL_ID = "gemini-3.8-flash"
```

Changes to settings such as `DOCUMENTS_DIR` need to be made before launching
the Python process. The configured directories are created automatically when
the app starts or receives an upload. Copy `.env.example` to `.env` and replace
the key placeholder with a valid key. `.env` is excluded from Git; never commit
or share it.
Crew tracing and Crew sharing are explicitly disabled for web requests, so the
terminal will not ask a Y/N trace-sharing question. This also prevents retrieved
document text from being uploaded to CrewAI as an execution trace.

Other processing parameters are code defaults:

| Parameter | Default | Where it is set |
|---|---:|---|
| Chunk size | 800 characters | `chunk_documents()` in `app/rag/chunking.py` |
| Chunk overlap | 120 characters | `chunk_documents()` in `app/rag/chunking.py` |
| Number of search candidates | 10 | `ask_question()` in `app/api/questions.py` |
| Number of final sources | 3 | `ask_question()` in `app/api/questions.py` |
| Maximum answer tokens per agent call | 350 | `generate_answer()` in `app/llm/generator.py` |
| Answer model | `gemini-3.8-flash` | `GEMINI_MODEL_ID` environment variable |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `app/rag/reranker.py` |

To change these defaults, edit the corresponding module. The API does not
currently accept chunk size, candidate count, model name, or generation length
as request parameters.

4. Start the API

From the project root:

```powershell
uvicorn app.main:app --reload
```

The server listens at `http://127.0.0.1:8000` by default.

Alternative: start via the CLI. This prints a summary of the configured CSV
when it exists, then starts the API:

```powershell
python -m app.main
```

Specify a different CSV or server address:

```powershell
python -m app.main --dataset "C:\data\medical.csv" --host 0.0.0.0 --port 8080
```

Print only the CSV summary and exit (does not start the API):

```powershell
python -m app.main --summary-only --dataset "data\medical.csv"
```

Evaluate the demonstration prediction model and print its metrics as JSON:

```powershell
python -m app.main --evaluate-model
```

This command trains the demo model on a fixed synthetic dataset, evaluates it
on a held-out 20% test split, prints the metrics, and exits without starting
the API. The test data is not used for fitting the model.

The compatibility script can also be used:

```powershell
python untitled129.py --summary-only
```

5. How to use the endpoints

Interactive API documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

Check that the server is running

Request:

```http
GET /health
```

Example response:

```json
{"status": "ok"}
```

 Upload a CSV, DOCX, PDF, or image

Upload one file at a time using `POST /api/documents`. Supported extensions are
`.csv`, `.docx`, `.pdf`, `.png`, `.jpg`, and `.jpeg`.

In Swagger UI, expand **POST `/api/documents`**, choose **Try it out**, select
the file in the `file` field, and click **Execute**.

PowerShell example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents" `
  -F "file=@C:\path\to\report.docx"
```

CSV example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents" `
  -F "file=@C:\path\to\records.csv"
```

PDF example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents" `
  -F "file=@C:\path\to\report.pdf"
```

The `file` field is the only upload parameter. The original upload is saved
under `data/documents/` using a generated ID as its filename. Its extracted
text is split into chunks and kept in application memory for questions.

Example response:

```json
{
  "document_id": "generated-id",
  "filename": "report.docx",
  "pages": 1,
  "chunks": 2,
  "laboratory_results": []
}
```

The `laboratory_results` array is populated when the text matches the
extractor's expected pattern, for example:
`Hemoglobin 13.5 g/dL 12.0 - 16.0`.

Upload errors include:

| Status | Meaning |
|---:|---|
| `400` | Empty upload |
| `413` | File exceeds `MAX_UPLOAD_BYTES` |
| `415` | Extension is not supported |
| `422` | File could not be read or contains no usable text |

CSV uploads are converted to text with column names and row values. DOCX
paragraphs and table rows are extracted. PDFs first use embedded text; pages
without enough embedded text fall back to OCR. Image uploads use OCR directly.
OCR requires the PaddleOCR dependencies and may download model assets on first
use.

# Ask a question about uploaded documents

Upload a document first, then use `POST /api/questions`.

Request JSON:

```json
{
  "question": "What hemoglobin value and reference range are reported?"
}
```

The `question` parameter is required, must contain 1–2000 characters, and is
sent in the JSON body. Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/questions" `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"What hemoglobin value and reference range are reported?\"}'
```

Example response shape:

```json
{
  "question": "What hemoglobin value and reference range are reported?",
  "answer": "The report gives a value of 13.5 g/dL [Source 1].",
  "sources": [
    {
      "source_label": "Source 1",
      "document_id": "generated-id",
      "page": 1,
      "chunk_id": "generated-id_p1_c0",
      "text": "Hemoglobin 13.5 g/dL 12.0 - 16.0"
    }
  ]
}
```

The retrieved text is sent to a three-agent CrewAI pipeline. The agents run
sequentially: (1) an evidence analyst extracts relevant details, (2) a verifier
checks support and citations against the passages, and (3) a response writer
produces the final answer. Each agent uses the Gemini model configured by
`GEMINI_MODEL_ID` (default `gemini-3.8-flash`), so one question can make several
Gemini calls. The server needs internet access and a valid `GEMINI_API_KEY`.
The first question may also download the cross-encoder reranker. If the key is
missing, CrewAI is not installed, or a Gemini request fails, the endpoint
returns a structured JSON error response instead of an unhandled server error.
If Gemini reports `503 UNAVAILABLE` or temporary high demand, the API response
is HTTP 503 with a concise detail and `retryable: true`, so the web client can
show the message and offer retry.
If Gemini reports `429 RESOURCE_EXHAUSTED` or an exhausted quota, the crew
stops processing and the API returns HTTP 429 with `retryable: false`. No
further agents run for that failed request.

Example temporary-overload response:

```json
{
  "error": "gemini_temporarily_unavailable",
  "detail": "Gemini is currently experiencing high demand. Please try again shortly.",
  "retryable": true
}
```

Example quota-exhausted response:

```json
{
  "error": "gemini_quota_exhausted",
  "detail": "Gemini quota is exhausted. Please check your API quota or billing.",
  "retryable": false
}
```

To check that the application loaded the key without printing the secret, run
this from the project root:

```powershell
python -c "import os; from app import config; print('Gemini key configured:', bool(os.getenv('GEMINI_API_KEY')))"
```

It should print `Gemini key configured: True`. Then restart the API after
changing `.env`, upload a document, and send a question. A successful
`GET /health` only confirms that the API is running; it does not verify the
Gemini key or model access.

Current implementation notes:

- The question endpoint searches the chunks uploaded during the current server
  process. Restarting the server clears this in-memory index; upload the
  documents again after a restart.
- Search currently uses BM25 lexical ranking, followed by the optional
  cross-encoder reranker. Although embedding and Qdrant helper modules exist,
  they are not currently wired into this endpoint.
- There is no delete-document endpoint yet. Uploaded source files remain in
  `data/documents/` until you remove them yourself.
- The BM25 index and reranker model are cached between questions in the same
  server process to reduce repeated retrieval setup time. The three CrewAI
  agents still make separate Gemini calls for each question.

### Get a demonstration prediction

`POST /api/predictions` accepts exactly four finite numeric values in
`features`:

```json
{
  "features": [0.8, -0.3, 0.5, 0.1]
}
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/predictions" `
  -H "Content-Type: application/json" `
  -d '{\"features\":[0.8,-0.3,0.5,0.1]}'
```

Example response:

```json
{"score": 0.72}
```

The score is a synthetic demo model's probability-like output between 0 and 1.
It is not a prediction for an actual patient's diagnosis or outcome. The
endpoint trains this demonstration model on synthetic data the first time it is
called; it does not read `data/medical.csv`.

### Evaluate the demonstration prediction model

You can inspect the same evaluation through `GET /api/predictions/evaluation`:

```powershell
curl.exe "http://127.0.0.1:8000/api/predictions/evaluation"
```

The response includes accuracy, precision, recall, F1 score, ROC-AUC, a
confusion matrix, per-class classification metrics, feature importance, and
the train/test split details. The confusion matrix uses this order:
`[[true_negative, false_positive], [false_negative, true_positive]]`.

These metrics measure performance only on the generated synthetic demo data.
They are not evaluation results for the healthcare CSV and do not show medical
accuracy or suitability for real patient data.

## 6. CSV summary output

The summary command reads `DATASET_PATH` (by default `data/medical.csv`) and
prints JSON containing:

- row and column counts
- column names and inferred data types
- missing-value counts
- the first five rows
- medical-condition distribution, if that column exists
- test-result distribution, if that column exists

It does not train a clinical model. To summarize another CSV:

```powershell
python -m app.main --summary-only --dataset "C:\data\another.csv"
```

## 7. Run tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover CSV and DOCX extraction, chunking/search, lab parsing, and
question/citation validation.

## 8. Docker

Build and start the container from the project root:

```powershell
docker build -t medical-ai-rag .
docker run --rm -p 8000:8000 medical-ai-rag
```

The Docker image copies the `data/` directory into the container. Uploaded
documents and in-memory search data are not persisted outside the container
unless you mount a host directory. For persistent uploads, for example:

```powershell
docker run --rm -p 8000:8000 `
  -v "C:\medical-ai-data:/app/data" `
  medical-ai-rag
```

The Docker image installs the full dependency list, including large ML/OCR
packages. Consider trimming `requirements.txt` for a lightweight deployment
that does not need every optional feature. The container does not include your
local `.env`; pass settings securely when starting it, for example with
`--env-file .env` (do not publish that file).

## 9. Troubleshooting

- **`Upload a document before asking a question`**: upload a document to
  `/api/documents` first, in the same running server session.
- **`DOCX ingestion requires python-docx`**: install dependencies with
  `pip install -r requirements.txt`.
- **PDF import errors**: ensure `pymupdf` is installed.
- **Image/scanned PDF OCR errors**: ensure Pillow, PaddleOCR, and the compatible
  PaddlePaddle package are installed.
- **`GEMINI_API_KEY` is missing**: confirm `.env` exists in the project root
  and contains the key assignment, then restart the app. Do not include quotes
  around the value unless they are part of the key.
- **Gemini API error or unsupported model**: confirm the key is active, has
  Gemini API access, and `GEMINI_MODEL_ID` is a model ID available to your
  account.
- **Slow first question or reranker download failure**: allow internet access
  to the model hub; the cross-encoder is downloaded separately from Gemini.
- **CSV not found in summary mode**: pass `--dataset` or set `DATASET_PATH` to
  the file's location.
