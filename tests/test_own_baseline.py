from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import ANSWER_SYSTEM, AnswerResponse
import own_baseline
from run_benchmark import ANSWER_SYSTEM as RUNNER_ANSWER_SYSTEM
from run_benchmark import dispatch_answering_llm_call


def test_components_use_shared_answer_system(monkeypatch, tmp_path):
    captured = {}

    class Index:
        def as_retriever(self, **kwargs):
            captured["retriever_options"] = kwargs
            return object()

    class StorageContext:
        @staticmethod
        def from_defaults(**kwargs):
            return kwargs

    class LLM:
        def as_structured_llm(self, output_cls):
            captured["output_cls"] = output_cls
            return object()

    def openai(**kwargs):
        captured["llm_options"] = kwargs
        return LLM()

    monkeypatch.setattr(own_baseline, "INDEX", tmp_path)
    monkeypatch.setattr(own_baseline, "OpenAIEmbedding", lambda **kwargs: object())
    monkeypatch.setattr(own_baseline, "OpenAI", openai)
    monkeypatch.setattr(own_baseline, "StorageContext", StorageContext)
    monkeypatch.setattr(own_baseline, "load_index_from_storage", lambda *args, **kwargs: Index())
    own_baseline._components.cache_clear()

    own_baseline._components()

    assert captured["llm_options"]["system_prompt"] is RUNNER_ANSWER_SYSTEM
    assert captured["output_cls"] is AnswerResponse
    assert captured["retriever_options"] == {"similarity_top_k": 5}
    own_baseline._components.cache_clear()


def test_own_baseline_retrieves_with_latest_message_and_passes_full_history(monkeypatch):
    expected = AnswerResponse(answer="It applies.", references=["Article 2"])
    captured = {}

    class Retriever:
        def retrieve(self, prompt):
            captured["retrieval_prompt"] = prompt
            return [type("Node", (), {"text": "Retrieved Act excerpt."})()]

    class LLM:
        def chat(self, messages):
            captured["messages"] = messages
            return type("Response", (), {"raw": expected})()

    monkeypatch.setattr(own_baseline, "_components", lambda: (Retriever(), LLM()))

    result = own_baseline.ask_own_baseline(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ]
    )

    assert result == expected
    assert captured["retrieval_prompt"] == "latest question"
    assert [(message.role.value, message.content) for message in captured["messages"]] == [
        ("user", "old question"),
        ("assistant", "old answer"),
        ("user", "Sources:\nRetrieved Act excerpt.\n\nlatest question"),
    ]
    assert own_baseline.ANSWER_SYSTEM is ANSWER_SYSTEM is RUNNER_ANSWER_SYSTEM
    assert dispatch_answering_llm_call("rag") is own_baseline.ask_own_baseline


def test_fresh_rag_index_builds_from_bundled_pdf_and_reloads(monkeypatch, tmp_path):
    from llama_index.core.embeddings import MockEmbedding

    class LLM:
        def as_structured_llm(self, output_cls):
            return self

    monkeypatch.setattr(own_baseline, "INDEX", tmp_path / "index")
    monkeypatch.setattr(own_baseline, "OpenAIEmbedding", lambda **kwargs: MockEmbedding(embed_dim=8))
    monkeypatch.setattr(own_baseline, "OpenAI", lambda **kwargs: LLM())
    own_baseline._components.cache_clear()
    try:
        retriever, _ = own_baseline._components()
        nodes = retriever.retrieve("What is a high-risk AI system?")
        assert len(nodes) == 5
        assert all(node.text.strip() for node in nodes)
        assert (own_baseline.INDEX / "docstore.json").is_file()

        # A later process must load the persisted index without rebuilding it.
        own_baseline._components.cache_clear()

        def no_rebuild(*args, **kwargs):
            raise AssertionError("The existing index should have been reused")

        monkeypatch.setattr(own_baseline, "PdfReader", no_rebuild)
        retriever, _ = own_baseline._components()
        assert len(retriever.retrieve("What is a high-risk AI system?")) == 5
    finally:
        own_baseline._components.cache_clear()
