"""Offline checks: python -m unittest test_knowledge_graph.py."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from RAG_LangFuse_KG import GraphRAG, expand_neighbors, extract_graph, parse_triples


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.document = SimpleNamespace(
            page_content="COPD is associated with cough.",
            metadata={"source": "guideline.pdf", "page": 2})
        self.fact = {"subject": "COPD", "subject_type": "Disease",
                     "relation": "has_symptom", "object": "cough",
                     "object_type": "Symptom",
                     "evidence": "COPD is associated with cough."}
        self.response = SimpleNamespace(content=json.dumps({"triples": [self.fact]}))

    def test_provenance_is_from_document(self):
        payload = {"triples": [dict(self.fact, page=999, source="invented.pdf")]}
        fact = parse_triples(json.dumps(payload), self.document)[0]
        self.assertEqual((fact["source"], fact["page"]), ("guideline.pdf", 3))

    def test_rejects_unsupported_evidence_and_invalid_schema(self):
        for change in ({"evidence": "COPD is cured by treatment."},
                       {"relation": "invented"}, {"object": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_triples(json.dumps({"triples": [dict(self.fact, **change)]}),
                              self.document)

    def test_checkpoint_resume_and_invalidation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            llm = Mock()
            llm.invoke.side_effect = [self.response, RuntimeError("offline")]
            with self.assertRaises(RuntimeError):
                extract_graph([self.document] * 2, llm, path, {"model": "a"}, {})
            self.assertEqual(len(json.loads(path.read_text())["chunks"]), 1)
            llm = Mock()
            llm.invoke.return_value = self.response
            graph = extract_graph([self.document] * 2, llm, path, {"model": "a"}, {})
            self.assertEqual(llm.invoke.call_count, 1)
            self.assertEqual(len(graph["triples"]), 1)
            self.assertEqual(len(graph["entities"]), 2)
            self.assertTrue(graph["complete"])
            extract_graph([self.document] * 2, llm, path, {"model": "a"}, {})
            self.assertEqual(llm.invoke.call_count, 1)
            extract_graph([self.document], llm, path, {"model": "b"}, {})
            self.assertEqual(llm.invoke.call_count, 2)

    def test_invalid_output_is_recorded_and_retried_on_next_run(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            llm = Mock()
            llm.invoke.return_value = SimpleNamespace(content="invalid json")
            graph = extract_graph([self.document], llm, path, {}, {})
            self.assertEqual(graph["chunks"], {})
            self.assertIn("0", graph["failed_chunks"])
            self.assertFalse(graph["complete"])
            self.assertEqual(llm.invoke.call_count, 3)
            llm.invoke.return_value = self.response
            graph = extract_graph([self.document], llm, path, {}, {})
            self.assertTrue(graph["complete"])
            self.assertEqual(graph["failed_chunks"], {})
            self.assertEqual(llm.invoke.call_count, 4)

    def test_schema_format_variants_are_normalized(self):
        fact = dict(self.fact, subject_type=" disease ", relation="Has Symptom")
        parsed = parse_triples(json.dumps({"triples": [fact]}), self.document)[0]
        self.assertEqual(parsed["subject_type"], "Disease")
        self.assertEqual(parsed["relation"], "has_symptom")

    def test_retry_includes_validation_feedback(self):
        with TemporaryDirectory() as directory:
            llm = Mock()
            bad = dict(self.fact, subject_type="UnknownType")
            llm.invoke.side_effect = [
                SimpleNamespace(content=json.dumps({"triples": [bad]})), self.response,
            ]
            graph = extract_graph([self.document], llm, Path(directory) / "graph.json", {}, {})
            self.assertTrue(graph["complete"])
            messages = llm.invoke.call_args.args[0]
            self.assertIn("Invalid subject_type 'UnknownType'", messages[-1]["content"])

    def test_failed_chunk_does_not_stop_later_chunks(self):
        with TemporaryDirectory() as directory:
            llm = Mock()
            llm.invoke.side_effect = [SimpleNamespace(content="invalid")] * 3 + [self.response]
            graph = extract_graph([self.document] * 2, llm,
                                  Path(directory) / "graph.json", {}, {})
            self.assertIn("0", graph["failed_chunks"])
            self.assertIn("1", graph["chunks"])
            self.assertFalse(graph["complete"])
            self.assertEqual(len(graph["triples"]), 1)

    def test_expansion_is_one_hop_and_bounded(self):
        triples = [{"subject": "A", "object": "B"},
                   {"subject": "b", "object": "C"},
                   {"subject": "C", "object": "D"},
                   {"subject": "X", "object": "Y"}]
        self.assertEqual(expand_neighbors(triples, [0], 4), triples[:2])
        self.assertEqual(expand_neighbors(triples, [0], 1), triples[:1])
        self.assertEqual(expand_neighbors(triples, [], 4), [])

    def test_answer_receives_passage_and_graph_evidence(self):
        retriever, graph_store, llm = Mock(), Mock(), Mock()
        retriever.invoke.return_value = [self.document]
        graph_store.similarity_search.return_value = [SimpleNamespace(metadata={"fact_id": 0})]
        llm.invoke.return_value = SimpleNamespace(content="Answer [Page 3]")
        facts = parse_triples(self.response.content, self.document)
        pipeline = GraphRAG(retriever, graph_store, facts, llm, {}, 12)
        result = pipeline.rag_bot("What symptoms?")
        prompt = llm.invoke.call_args.args[0][0]["content"]
        self.assertIn("<passages>\n[Page 3]", prompt)
        self.assertIn("<graph>\n[Page 3] (COPD, has_symptom, cough)", prompt)
        self.assertEqual(result["graph_triples"], facts)
        self.assertEqual(result["answer"], "Answer [Page 3]")
        pipeline.graph_store = None
        pipeline.triples = []
        self.assertEqual(pipeline.rag_bot("Question")["graph_triples"], [])


if __name__ == "__main__":
    unittest.main()
