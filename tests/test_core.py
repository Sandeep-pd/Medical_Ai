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
from app.llm.safety import validate_question, validate_source_references
from app.ml.predictor import evaluate_demo_model
from app.llm.crew import (
    CrewAIAnswerError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
    GEMINI_QUOTA_EXHAUSTED_MESSAGE,
    TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE,
    run_medical_crew,
)
from app.llm.generator import generate_answer


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
                patch("app.api.questions.rerank_results", return_value=documents_api.indexed_chunks),
                patch(
                    "app.api.questions.generate_answer",
                    side_effect=CrewAIAnswerError(
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
            "error": "crewai_answer_error",
            "detail": "Gemini API request failed.",
            "retryable": False,
        })

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
                patch("app.api.questions.rerank_results", return_value=documents_api.indexed_chunks),
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
                patch("app.api.questions.rerank_results", return_value=documents_api.indexed_chunks),
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

    def test_crew_converts_gemini_503_to_short_message(self):
        crewai_stub = MagicMock()
        crewai_stub.Agent.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        crewai_stub.Task.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        crewai_stub.Process.sequential = "sequential"
        crewai_stub.Crew.return_value.kickoff.side_effect = RuntimeError(
            "503 UNAVAILABLE: This model is currently experiencing high demand."
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(sys.modules, {"crewai": crewai_stub}),
        ):
            with self.assertRaises(GeminiUnavailableError) as error:
                run_medical_crew("Question?", "Evidence.")

        self.assertEqual(str(error.exception), TEMPORARY_GEMINI_UNAVAILABLE_MESSAGE)

    def test_crew_converts_gemini_quota_error_and_stops(self):
        crewai_stub = MagicMock()
        crewai_stub.Agent.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        crewai_stub.Task.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        crewai_stub.Process.sequential = "sequential"
        crewai_stub.Crew.return_value.kickoff.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED: You exceeded your current quota."
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(sys.modules, {"crewai": crewai_stub}),
        ):
            with self.assertRaises(GeminiQuotaExceededError) as error:
                run_medical_crew("Question?", "Evidence.")

        self.assertEqual(str(error.exception), GEMINI_QUOTA_EXHAUSTED_MESSAGE)
        crewai_stub.Crew.return_value.kickoff.assert_called_once()

    def test_three_agent_crew_uses_configured_gemini_model_and_context(self):
        created_agents = []
        created_tasks = []
        created_crews = []

        def make_agent(**kwargs):
            agent = SimpleNamespace(**kwargs)
            created_agents.append(agent)
            return agent

        def make_task(**kwargs):
            task = SimpleNamespace(**kwargs)
            created_tasks.append(task)
            return task

        class StubCrew:
            def __init__(self, **kwargs):
                self.options = kwargs
                created_crews.append(self)

            def kickoff(self, inputs):
                self.inputs = inputs
                return SimpleNamespace(raw="Supported answer [Source 1].")

        crewai_stub = SimpleNamespace(
            Agent=make_agent,
            Crew=StubCrew,
            LLM=lambda **kwargs: SimpleNamespace(**kwargs),
            Process=SimpleNamespace(sequential="sequential"),
            Task=make_task,
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(sys.modules, {"crewai": crewai_stub}),
        ):
            answer = run_medical_crew("Question?", "Evidence.")

        self.assertEqual(answer, "Supported answer [Source 1].")
        self.assertEqual(
            created_agents[0].llm.model,
            "gemini/gemini-3.8-flash",
        )
        self.assertEqual(len(created_agents), 3)
        self.assertEqual(len(created_tasks), 3)
        self.assertIn("{context}", created_tasks[1].description)
        self.assertEqual(created_crews[0].options["process"], "sequential")
        self.assertFalse(created_crews[0].options["tracing"])
        self.assertFalse(created_crews[0].options["share_crew"])
        self.assertEqual(
            created_crews[0].inputs,
            {"question": "Question?", "context": "Evidence."},
        )

    def test_three_agent_crew_requires_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                run_medical_crew("Question?", "Evidence.")

    def test_answer_generator_uses_three_agent_crew(self):
        with patch(
            "app.llm.generator.run_medical_crew",
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
