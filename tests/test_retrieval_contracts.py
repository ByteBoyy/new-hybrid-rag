import unittest

from pydantic import ValidationError

from src.models import QueryRequest
from src.retrieval.hybrid_retriever import HybridRetriever


class HybridRetrieverContractTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever.__new__(HybridRetriever)
        self.retriever.cfg = {
            "semantic_weight": 1.0,
            "lexical_weight": 0.3,
        }

    def test_normalize_scores_does_not_mutate_input(self):
        results = [{"id": "a", "score": 2.0}, {"id": "b", "score": 4.0}]

        normalized = self.retriever._normalize_scores(results)

        self.assertEqual([2.0, 4.0], [item["score"] for item in results])
        self.assertEqual([0.0, 1.0], [item["score"] for item in normalized])

    def test_normalize_equal_scores_returns_one_for_each(self):
        results = [{"id": "a", "score": 3.0}, {"id": "b", "score": 3.0}]

        normalized = self.retriever._normalize_scores(results)

        self.assertEqual([1.0, 1.0], [item["score"] for item in normalized])

    def test_fusion_keeps_lexical_only_documents(self):
        semantic = [{"id": "a", "score": 0.8, "title": "semantic"}]
        lexical = [
            {"id": "a", "score": 1.0, "title": "semantic"},
            {"id": "b", "score": 0.5, "title": "lexical"},
        ]

        fused = self.retriever._fuse_results(semantic, lexical, top_k=2)

        self.assertEqual(["a", "b"], [item["id"] for item in fused])
        self.assertEqual(0.0, fused[1]["sem_score"])
        self.assertEqual("lexical", fused[1]["title"])


class QueryRequestContractTests(unittest.TestCase):
    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValidationError):
            QueryRequest(query="photosynthesis", mode="unsupported")

    def test_accepts_supported_modes(self):
        for mode in ("standard", "ultra"):
            self.assertEqual(mode, QueryRequest(query="photosynthesis", mode=mode).mode)


if __name__ == "__main__":
    unittest.main()
