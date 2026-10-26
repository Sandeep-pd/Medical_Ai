"""Focused tests for core ingestion helpers."""

import unittest
import csv
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.main import app
from app.ingestion.lab_extractor import extract_lab_results
from app.ingestion.layout_parser import ingest_document
from app.rag.chunking import chunk_documents
from app.rag.hybrid_search import _get_bm25_index, hybrid_search
from app.rag.indexing import collection_count, index_chunks, search_chroma
from app.llm.safety import validate_question, validate_source_references
from app.ml.predictor import evaluate_demo_model
from app.llm.crew import (
    GeminiAnswerError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
    GEMINI_QUOTA_EXHAUSTED_MESSAGE,
    TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE,
    generate_medical_answer,
)
from app.llm.generator import generate_answer


def make_genai_stub(generate_content):
    types_stub = SimpleNamespace(
        HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        HttpRetryOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    genai_stub = SimpleNamespace(Client=MagicMock(return_value=client))
    google_stub = SimpleNamespace(genai=genai_stub)
    return google_stub, SimpleNamespace(types=types_stub), genai_stub, client


class CorePipelineTests(unittest.TestCase):
    def test_extract_lab_result(self):
        result = extract_lab_results([{
            "page": 2,
            "method": "pdf_text",
            "text": "Hemoglobin 13.5 g/dL 12.0 - 16.0",
        }])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["test_name"], "Hemoglobin")
        self.assertEqual(result[0]["page"], 2)

    def test_chunk_and_retrieve(self):
        chunks = chunk_documents(
            [{"page": 1, "method": "pdf_text", "text": "Hemoglobin result is documented. " * 30}],
            document_id="doc1",
            chunk_size=80,
            chunk_overlap=10,
        )
        self.assertTrue(chunks)
        self.assertEqual(hybrid_search("hemoglobin", chunks, top_k=1)[0]["document_id"], "doc1")

    def test_bm25_index_is_reused_for_same_document_chunks(self):
        chunks = [{"text": "hemoglobin result"}, {"text": "glucose value"}]
        _get_bm25_index.cache_clear()

        hybrid_search("hemoglobin", chunks, top_k=1)
        hybrid_search("glucose", chunks, top_k=1)

        self.assertEqual(_get_bm25_index.cache_info().hits, 1)
        _get_bm25_index.cache_clear()

    def test_hybrid_search_fuses_lexical_and_dense_results(self):
        chunks = [
            {"chunk_id": "lexical", "text": "hemoglobin levels"},
            {"chunk_id": "dense", "text": "blood test results"},
        ]
        dense_results = [
            {**chunks[1], "score": 0.9},
            {**chunks[0], "score": 0.2},
        ]

        results = hybrid_search("hemoglobin", chunks, top_k=2, dense_results=dense_results)

        self.assertEqual({item["chunk_id"] for item in results}, {"lexical", "dense"})
        self.assertGreater(results[0]["score"], 0)

    def test_chroma_index_persists_and_returns_semantic_results(self):
        class FakeVector:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        class FakeVectors(FakeVector):
            def __getitem__(self, index):
                return FakeVector(self.values[index])

        model = MagicMock()
        model.encode.side_effect = lambda texts, **kwargs: FakeVectors(
            [[1.0, 0.0] for _ in texts]
        )
        collection = MagicMock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "ids": [["doc_p1_c0"]],
            "documents": [["Hemoglobin result is normal."]],
            "metadatas": [[{"document_id": "doc", "page": 1}]],
            "distances": [[0.05]],
        }
        client = MagicMock()
        client.get_or_create_collection.return_value = collection
        chromadb_stub = SimpleNamespace(PersistentClient=MagicMock(return_value=client))
        chunks = [{
            "chunk_id": "doc_p1_c0",
            "document_id": "doc",
            "page": 1,
            "text": "Hemoglobin result is normal.",
        }]

        with (
            patch.dict(sys.modules, {"chromadb": chromadb_stub}),
            patch("app.rag.indexing.config.CHROMA_DIR", Path("data/chroma")),
            patch("app.rag.indexing.get_embedding_model", return_value=model),
        ):
            index_chunks(chunks)
            self.assertEqual(1, collection_count())
            results = search_chroma("hemoglobin", top_k=1)

        chromadb_stub.PersistentClient.assert_called_with(path="data\\chroma")
        collection.upsert.assert_called_once()
        self.assertEqual(results[0]["chunk_id"], "doc_p1_c0")
        self.assertEqual(results[0]["text"], "Hemoglobin result is normal.")
        self.assertEqual(results[0]["document_id"], "doc")
        self.assertGreater(results[0]["score"], 0.9)

    def test_question_and_citation_validation(self):
        self.assertFalse(validate_question("  ")[0])
        citations = validate_source_references(
            "This is supported [Source 1].",
            [{"source_label": "Source 1"}],
        )
        self.assertTrue(citations["valid"])

    def test_ingest_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Patient", "Result"])
                writer.writerow(["Sample", "Normal"])

            documents = ingest_document(path)

        self.assertEqual(documents[0]["method"], "csv_text")
        self.assertIn("Patient: Sample", documents[0]["text"])
        self.assertIn("Result: Normal", documents[0]["text"])

    def test_ingest_docx_paragraph_and_table(self):
        try:
            from docx import Document
        except ImportError:
            self.skipTest("python-docx is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            document = Document()
            document.add_paragraph("Patient report")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Test"
            table.cell(0, 1).text = "Normal"
            document.save(path)

            documents = ingest_document(path)

        self.assertEqual(documents[0]["method"], "docx_text")
        self.assertIn("Patient report", documents[0]["text"])
        self.assertIn("Test | Normal", documents[0]["text"])

    def test_demo_model_evaluation_reports_holdout_metrics(self):
        evaluation = evaluate_demo_model()

        self.assertEqual(evaluation["training_samples"], 1600)
        self.assertEqual(evaluation["test_samples"], 400)
        self.assertGreaterEqual(evaluation["accuracy"], 0.0)
        self.assertLessEqual(evaluation["accuracy"], 1.0)
        self.assertGreaterEqual(evaluation["roc_auc"], 0.0)
        self.assertEqual(len(evaluation["confusion_matrix"]), 2)
        self.assertIn("feature_1", evaluation["feature_importance"])

    def test_question_model_initialization_error_returns_503(self):
        documents_api.indexed_chunks[:] = [{
            "chunk_id": "doc_p1_c0",
            "document_id": "doc",
            "page": 1,
            "text": "Test evidence",
        }]
        try:
            with (
                patch("app.api.questions.hybrid_search", return_value=documents_api.indexed_chunks),
                patch("app.api.questions.search_chroma", return_value=[]),
                patch(
                    "app.api.questions.generate_answer",
                    side_effect=GeminiAnswerError(
                        "Gemini API request failed."
                    ),
                ),
            ):
                response = TestClient(app).post(
                    "/api/questions",
                    json={"question": "What does the document say?"},
                )
        finally:
            documents_api.indexed_chunks.clear()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "error": "gemini_answer_error",
            "detail": "Gemini API request failed.",
            "retryable": False,
        })

    def test_question_endpoint_accepts_valid_json_and_returns_sources(self):
        documents_api.indexed_chunks[:] = [{
            "chunk_id": "doc_p1_c0",
            "document_id": "doc",
            "page": 1,
            "text": "The report says the patient had a cough.",
        }]
        try:
            with (
                patch("app.api.questions.search_chroma", return_value=[]),
                patch(
                    "app.api.questions.hybrid_search",
                    return_value=documents_api.indexed_chunks,
                ),
                patch(
                    "app.api.questions.generate_answer",
                    return_value="The report notes a cough [Source 1].",
                ),
            ):
                response = TestClient(app).post(
                    "/api/questions",
                    json={"question": "What symptom does the report mention?"},
                )
        finally:
            documents_api.indexed_chunks.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The report notes a cough [Source 1].")
        self.assertEqual(response.json()["sources"][0]["chunk_id"], "doc_p1_c0")

    def test_gemini_unavailable_returns_only_friendly_503_message(self):
        documents_api.indexed_chunks[:] = [{
            "chunk_id": "doc_p1_c0",
            "document_id": "doc",
            "page": 1,
            "text": "Test evidence",
        }]
        try:
            with (
                patch("app.api.questions.hybrid_search", return_value=documents_api.indexed_chunks),
                patch("app.api.questions.search_chroma", return_value=[]),
                patch(
                    "app.api.questions.generate_answer",
                    side_effect=GeminiUnavailableError(
                        TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE
                    ),
                ),
            ):
                response = TestClient(app).post(
                    "/api/questions",
                    json={"question": "What does the document say?"},
                )
        finally:
            documents_api.indexed_chunks.clear()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "error": "gemini_temporarily_unavailable",
            "detail": TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE,
            "retryable": True,
        })

    def test_gemini_quota_exhausted_returns_short_429_message(self):
        documents_api.indexed_chunks[:] = [{
            "chunk_id": "doc_p1_c0",
            "document_id": "doc",
            "page": 1,
            "text": "Test evidence",
        }]
        try:
            with (
                patch("app.api.questions.hybrid_search", return_value=documents_api.indexed_chunks),
                patch("app.api.questions.search_chroma", return_value=[]),
                patch(
                    "app.api.questions.generate_answer",
                    side_effect=GeminiQuotaExceededError(
                        GEMINI_QUOTA_EXHAUSTED_MESSAGE
                    ),
                ) as generate,
            ):
                response = TestClient(app).post(
                    "/api/questions",
                    json={"question": "What does the document say?"},
                )
        finally:
            documents_api.indexed_chunks.clear()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {
            "error": "gemini_quota_exhausted",
            "detail": GEMINI_QUOTA_EXHAUSTED_MESSAGE,
            "retryable": False,
        })
        generate.assert_called_once()

    def test_gemini_request_converts_503_to_short_message(self):
        error = RuntimeError(
            "Google Gemini API error: 503 - This model is currently experiencing "
            "high demand. Spikes in demand are usually temporary. Please try again later."
        )
        generate_content = MagicMock(side_effect=[error] * 3)
        google_stub, types_stub, _, _ = make_genai_stub(generate_content)
        with (
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL_IDS": (
                    "models/gemini-3.5-flash-lite,"
                    "models/gemini-3.5-flash,"
                    "models/gemini-3.6-flash"
                ),
            }),
            patch.dict(sys.modules, {"google": google_stub, "google.genai": types_stub}),
        ):
            with self.assertRaises(GeminiUnavailableError) as error:
                generate_medical_answer("Question?", "Evidence.")

        self.assertEqual(str(error.exception), TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE)
        self.assertEqual(
            [call.kwargs["model"] for call in generate_content.call_args_list],
            [
                "models/gemini-3.5-flash-lite",
                "models/gemini-3.5-flash",
                "models/gemini-3.6-flash",
            ],
        )

    def test_gemini_request_converts_504_to_short_message(self):
        error = RuntimeError(
            "504 DEADLINE_EXCEEDED: Deadline expired before operation could complete."
        )
        generate_content = MagicMock(side_effect=[error] * 3)
        google_stub, types_stub, _, _ = make_genai_stub(generate_content)
        with (
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL_IDS": (
                    "models/gemini-3.5-flash-lite,"
                    "models/gemini-3.5-flash,"
                    "models/gemini-3.6-flash"
                ),
            }),
            patch.dict(sys.modules, {"google": google_stub, "google.genai": types_stub}),
        ):
            with self.assertRaises(GeminiUnavailableError) as error:
                generate_medical_answer("Question?", "Evidence.")

        self.assertEqual(str(error.exception), TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE)

    def test_gemini_request_converts_quota_error(self):
        error = RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota.")
        generate_content = MagicMock(side_effect=[error] * 3)
        google_stub, types_stub, _, _ = make_genai_stub(generate_content)
        with (
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL_IDS": (
                    "models/gemini-3.5-flash-lite,"
                    "models/gemini-3.5-flash,"
                    "models/gemini-3.6-flash"
                ),
            }),
            patch.dict(sys.modules, {"google": google_stub, "google.genai": types_stub}),
        ):
            with self.assertRaises(GeminiQuotaExceededError) as error:
                generate_medical_answer("Question?", "Evidence.")

        self.assertEqual(str(error.exception), GEMINI_QUOTA_EXHAUSTED_MESSAGE)
        self.assertEqual(generate_content.call_count, 3)

    def test_gemini_answer_uses_one_bounded_safety_checked_request(self):
        generate_content = MagicMock(
            side_effect=[
                RuntimeError("model not found"),
                SimpleNamespace(text="Supported answer [Source 1]."),
            ]
        )
        google_stub, types_stub, genai_stub, client = make_genai_stub(generate_content)
        with (
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL_IDS": (
                    "models/gemini-3.5-flash-lite,"
                    "models/gemini-3.5-flash,"
                    "models/gemini-3.6-flash"
                ),
            }),
            patch.dict(sys.modules, {"google": google_stub, "google.genai": types_stub}),
        ):
            answer = generate_medical_answer("Question?", "Evidence.")

        self.assertEqual(answer, "Supported answer [Source 1].")
        genai_stub.Client.assert_called_once()
        client_options = genai_stub.Client.call_args.kwargs
        self.assertEqual(
            client_options["http_options"].timeout,
            10_000,
        )
        self.assertEqual(
            client_options["http_options"].retry_options.attempts,
            1,
        )
        self.assertEqual(
            [call.kwargs["model"] for call in generate_content.call_args_list],
            ["models/gemini-3.5-flash-lite", "models/gemini-3.5-flash"],
        )
        prompt = generate_content.call_args.kwargs["contents"]
        self.assertIn("general medical", prompt)
        self.assertIn("do not infer a diagnosis", prompt)
        self.assertIn("Question:\nQuestion?", prompt)
        self.assertIn("Retrieved passages:\nEvidence.", prompt)
        self.assertEqual(generate_content.call_args.kwargs["config"].max_output_tokens, 350)

    def test_gemini_answer_requires_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                generate_medical_answer("Question?", "Evidence.")

    def test_answer_generator_uses_grounded_answer_generation(self):
        with patch(
            "app.llm.generator.generate_medical_answer",
            return_value="Final answer.",
        ) as run_crew:
            answer = generate_answer("Question?", "Evidence.")

        self.assertEqual(answer, "Final answer.")
        run_crew.assert_called_once_with(
            question="Question?",
            context="Evidence.",
            max_output_tokens=350,
        )


if __name__ == "__main__":
    unittest.main()
