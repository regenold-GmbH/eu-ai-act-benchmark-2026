import json
import re
import sys
from types import SimpleNamespace

import pytest

import own_baseline
import run_benchmark as benchmark
from contracts import AnswerResponse
from summarize_results import summarize


@pytest.fixture
def providers(monkeypatch):
    calls = []

    def responses(**kwargs):
        schema = kwargs["text_format"]
        calls.append((schema.__name__, kwargs))
        if schema is AnswerResponse:
            result = AnswerResponse(answer="It applies.", references=["Article 2"])
        elif schema is benchmark.JudgeCorrectnessResponse:
            criteria = kwargs["input"][-1]["content"].split("# Correctness criteria:\n")[1]
            result = schema(criteria=[
                benchmark.CriterionJudgeResult(criterion=criterion, consideration="Mock judge", passed=True)
                for criterion in re.findall(r"^\d+\. (.+)$", criteria, flags=re.MULTILINE)
            ])
        else:
            result = schema(
                consideration="Mock judge", appropriateness_passed=True, clarity_passed=True,
            )
        return SimpleNamespace(output_text=result.model_dump_json())

    class Retriever:
        def retrieve(self, question):
            calls.append(("retrieval", {"question": question}))
            return [SimpleNamespace(text="EU AI Act source excerpt")]

    class RAGModel:
        def chat(self, messages):
            calls.append(("rag", {"messages": messages}))
            return SimpleNamespace(raw=AnswerResponse(
                answer="It applies.", references=["Article 2"],
            ))

    monkeypatch.setattr(benchmark.litellm, "responses", responses)
    monkeypatch.setattr(own_baseline, "_components", lambda: (Retriever(), RAGModel()))
    return calls


@pytest.mark.parametrize("method", ["web_search", "rag", "custom"])
@pytest.mark.parametrize("hard", [False, True])
def test_cli_evaluates_bundled_dataset_resumes_and_summarizes(
    monkeypatch, tmp_path, providers, method, hard,
):
    if method == "custom":
        (tmp_path / "research_method.py").write_text(
            "from contracts import AnswerResponse\n"
            "calls = []\n"
            "def answer(history):\n"
            "    calls.append(('custom', {'messages': history}))\n"
            "    return AnswerResponse(answer='It applies.', references=['Article 2'])\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "research_method", raising=False)
        selection = ["--method", "research_method:answer"]
    else:
        selection = ["--baseline", method]
    output = tmp_path / "results.json"
    args = [
        "run_benchmark.py", *selection,
        "--limit", "2", "--output", str(output),
        "--no_judgements", "3", "--judge_temperature", "0.1",
    ]
    if hard:
        args += ["--simulate_history", "--simulate_pushback"]
    monkeypatch.setattr(sys, "argv", args)
    benchmark.main()
    entries = json.loads(output.read_text())
    assert [row["row_index"] for row in entries] == [0, 1]
    assert all(len(row["judgements_correctness"]) == 3 for row in entries)
    assert all(row["criteria"] for row in entries)
    assert all(
        len(judgement["criteria"]) == len(row["criteria"])
        for row in entries for judgement in row["judgements_correctness"]
    )
    assert all(len(row["judgements_tone"]) == 3 for row in entries)
    config = entries[0]["run_config"]
    if method == "custom":
        assert config["method"] == "research_method:answer"
    else:
        assert config["baseline"] == method
    assert config["simulate_history"] is hard
    assert len(config["dataset_sha256"]) == 64
    if method == "rag":
        assert config["retrieval_top_k"] == 5
        assert len(config["pdf_sha256"]) == 64

    custom_calls = sys.modules["research_method"].calls if method == "custom" else []
    kind = {"web_search": "AnswerResponse", "rag": "rag", "custom": "custom"}[method]
    answer_calls = [options for name, options in providers + custom_calls if name == kind]
    assert len(answer_calls) == (4 if hard else 2)
    first_history = answer_calls[0].get("input", answer_calls[0].get("messages"))
    # OpenAI/LiteLLM histories include a system message; RAG sets it on its LLM.
    system_messages = 1 if method == "web_search" else 0
    assert len(first_history) == system_messages + (19 if hard else 1)
    if hard:
        last_history = answer_calls[1].get("input", answer_calls[1].get("messages"))
        assert len(last_history) == system_messages + 21
    if method == "web_search":
        assert all(call["tools"] == [{"type": "web_search_preview"}] for call in answer_calls)
    if method == "custom":
        assert all(message["role"] in {"user", "assistant"} for call in answer_calls for message in call["messages"])
    judge_calls = [options for name, options in providers if name.startswith("Judge")]
    assert len(judge_calls) == 12
    assert all(call["temperature"] == 0.1 for call in judge_calls)
    assert all(call["tools"] is None for call in judge_calls)

    calls_before_resume = len(providers) + len(custom_calls)
    benchmark.main()
    assert len(providers) + len(custom_calls) == calls_before_resume
    assert json.loads(output.read_text()) == entries

    # Increasing the selected range appends only the new row.
    args[args.index("--limit") + 1] = "3"
    benchmark.main()
    entries = json.loads(output.read_text())
    assert [row["row_index"] for row in entries] == [0, 1, 2]
    assert summarize(entries)["metrics"]["Answer correctness (strict)"]["mean"] == 100

    # A different judge configuration must not silently reuse prior scores.
    args[args.index("--judge_temperature") + 1] = "0.2"
    with pytest.raises(ValueError, match="Checkpoint configuration differs"):
        benchmark.main()
    args.append("--overwrite_output")
    benchmark.main()
    assert all(row["run_config"]["judge_temperature"] == 0.2 for row in json.loads(output.read_text()))


def test_checkpoint_survives_interrupted_run(monkeypatch, tmp_path, providers):
    original = benchmark.ask_web_search
    calls = 0

    def interrupted(history):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated provider failure")
        return original(history)

    output = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="Simulated provider failure"):
        benchmark.run(2, output, False, answering_llm_call=interrupted)
    assert [row["row_index"] for row in json.loads(output.read_text())] == [0]
    benchmark.run(2, output, False, answering_llm_call=original)
    assert [row["row_index"] for row in json.loads(output.read_text())] == [0, 1]


@pytest.mark.parametrize("selection", [
    [],
    ["--method", "example_method:answer", "--baseline", "web_search"],
    ["--method", "missing_separator"],
    ["--method", "module_that_does_not_exist:answer"],
    ["--method", "example_method:missing_function"],
    ["--method", "contracts:REGULATION_NAME"],
])
def test_invalid_method_selection_fails_before_model_calls(monkeypatch, selection):
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", *selection])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("hard", [False, True])
def test_example_method_runs_through_cli(monkeypatch, tmp_path, providers, hard):
    output = tmp_path / "mock.json"
    args = [
        "run_benchmark.py", "--method", "example_method:answer",
        "--limit", "1", "--output", str(output),
    ]
    if hard:
        args += ["--simulate_history", "--simulate_pushback"]
    monkeypatch.setattr(sys, "argv", args)
    benchmark.main()
    row, = json.loads(output.read_text())
    assert "mock response" in row["candidate_answer"]
    assert row["candidate_references"] == []
    assert row["run_config"]["method"] == "example_method:answer"
    assert summarize([row])["metrics"]


@pytest.mark.parametrize("options, message", [
    ({"no_judgements": 0}, "at least 1"),
    ({"no_judgements": 3}, "judge_temperature should be > 0"),
    ({"limit": 0}, "limit must be at least 1"),
    ({"specific_rows": [-1]}, "outside the dataset"),
])
def test_invalid_configuration_fails_before_model_calls(tmp_path, options, message):
    kwargs = {"limit": 2, "output_path": tmp_path / "invalid.json", "debug": False, **options}
    with pytest.raises(ValueError, match=message):
        benchmark.run(**kwargs)
