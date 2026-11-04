# regenold 2026 EU AI Act benchmark

A 110-question, expert-annotated benchmark for regulatory question answering
about Regulation (EU) 2024/1689, evaluated in single-turn (easy) and adversarial
multi-turn (hard) settings. The evaluation scope is the EU AI Act as of
May 1, 2026.

The benchmark can be used to evaluate your own answering system with
questions, prompts, LLM judges, and scoring rules. It includes the annotated
workbook (`data/2026-05-11-regenold_challenge_v01_01.xlsx`), fixed conversation history, and EU AI Act source PDF. Private contestant integrations and historical results discussed in the respective paper are not distributed.

## Setup

Install Python 3.13+ and [uv](https://docs.astral.sh/uv/), then run from this
repository's root:

```bash
uv sync --frozen
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for the correctness and tone judge
(GPT-5.4 by default). Add any credentials and dependencies your own method
requires.

## Connect your method

```bash
cp src/example_method.py src/my_method.py
```

Replace the mock body in `my_method.py` with a call to your system. The runner
imports `--method my_method:answer` and calls this synchronous function:

```python
def answer(history: list[dict[str, str]]) -> AnswerResponse:
    ...
```

Import `AnswerResponse` from `contracts` and return an instance with:

- `answer`: the final answer text.
- `references`: a list such as `["Article 3", "Article 50.4", "Annex IV.2"]`.
- `reasoning`: optional reasoning text, excluded from scoring.

`history` contains user and assistant messages with `role` and `content` keys;
its last message is the current user request. Pass the full history to your
method. Your method can use any prompt; it only needs to return `AnswerResponse`
with the fields and citation format described above. `ANSWER_SYSTEM` in
`src/contracts.py` is the system prompt used by the built-in baselines and can
serve as an example.

## Evaluate

Here's how to test two questions and one judgement per answer for a smoke run:

```bash
uv run --frozen python src/run_benchmark.py \
  --method my_method:answer --limit 2 --output results/smoke.json
uv run --frozen python src/summarize_results.py results/smoke.json
```

For the full benchmark, we recommend three independent judgements repeats per answer at judge temperature 0.1, across both easy and hard modes. The following commands run the full benchmark and summarize the results:

```bash
# Easy: one user question per item.
uv run --frozen python src/run_benchmark.py \
  --method my_method:answer --no_judgements 3 --judge_temperature 0.1 \
  --output results/my-method-easy.json

# Hard: fixed nine-exchange history, then question, then adversarial pushback.
uv run --frozen python src/run_benchmark.py \
  --method my_method:answer --no_judgements 3 --judge_temperature 0.1 \
  --simulate_history --simulate_pushback --output results/my-method-hard.json

uv run --frozen python src/summarize_results.py \
  results/my-method-easy.json results/my-method-hard.json
```

Note: Easy mode calls your method once per question; hard mode calls it twice as it features
a user pushback.
Only the final response is scored in hard mode, including its latency.
The initial response and judging time are excluded from that latency.
Each question in hard mode starts with the same fixed history.

Results are checkpointed after every completed question. Rerun the same command
to resume; `--overwrite_output` starts over. Checkpoints record the method
identifier, judge settings, conversation flags, and input hashes, and reject
incompatible configurations. They cannot detect changes to your method's code,
models, or external service configuration: use a new output path after such
changes. Keep smoke runs separate from full runs with different judge settings.

Use `--specific_rows 0 4 10` for zero-based workbook rows, or `--limit N` for the
first N rows. Without `--output`, custom methods write `results/custom.json` in easy mode and
`results/custom-hard.json` in hard mode.

## Scores

The summarizer reports percentages (higher is better), with the minimum and
maximum across repeated judge runs in brackets:

| Metric                        | Definition                                                                 |
| ----------------------------- | -------------------------------------------------------------------------- |
| Answer correctness, loose     | Fraction of annotated correctness criteria passed                          |
| Answer correctness, strict    | Fraction of questions where every criterion passed, computed per judge run |
| Answer conciseness            | Mean of `100 / max(1, answer_words / reference_answer_words)`              |
| Reference correctness, loose  | Fraction of expected top-level Articles/Annexes cited                      |
| Reference correctness, strict | Fraction of expected Articles/Annexes cited, retaining the first sublevel  |
| Reference conciseness         | Mean of `100 / (1 + extra_strict_references / expected_strict_references)` |
| Regulatory tone               | Fraction judged both appropriate and clear                                 |
| Response speed                | Mean of `max(0, 100 - latency_seconds)`                                    |

The overall score is the geometric mean of the eight mean metric scores; a
zero metric produces zero overall. Its range is computed from each judge
run's overall score. Rows with no annotated expected references are excluded
from reference metrics; unavailable metrics are omitted from the overall score.
Ranges show judge variation, not statistical confidence intervals. Exact citation
matching rules are in `src/ref_scoring.py`.

## Optional baseline examples

Two built-in methods illustrate the adapter interface: GPT-5.5 with web search
and GPT-5.5 with RAG over the bundled PDF. Select one instead of `--method`:

```bash
uv run --frozen python src/run_benchmark.py --baseline web_search --limit 2
uv run --frozen --extra rag python src/run_benchmark.py --baseline rag --limit 2
```

Both use `OPENAI_API_KEY`; model overrides are in `.env.example`. RAG dependencies
are installed only with `--extra rag`. RAG uses `text-embedding-3-small`, default
LlamaIndex chunking, and five excerpts retrieved with the latest user message;
generation receives the full history. Its first run builds a local index at
`data/ai_act/own_baseline_index/` and incurs embedding calls. Later runs reuse it;
delete it after changing the PDF. No precomputed index is distributed.

## Tests and layout

```bash
uv run --frozen --extra rag pytest -q
```

Tests use mocked provider calls and require no credentials. They cover custom
method loading, both conversation settings, checkpoint resumption, the optional
baselines, and scoring.

```text
data/ai_act/              Workbook, fixed history, EU AI Act PDF
src/contracts.py         Answer schema and baseline answering prompt
src/example_method.py    Mock adapter to replace with your method
src/run_benchmark.py     Evaluation, judge prompts, checkpointing
src/*scoring.py          Deterministic reference and latency scoring
src/summarize_results.py Text score summaries
src/own_baseline/        Optional RAG example
tests/                   Offline tests
```

## License

© 2026 regenold GmbH. Code and benchmark annotations are licensed under
the [Apache License 2.0](LICENSE). The bundled EU legal text is a third-party
source document; see [NOTICE](NOTICE).

## Publication

If you use this benchmark, please cite our paper:

```
@misc{regenold2026euaiact,
  title={Are {LLM} Assistants Regulatory Ready? Answers from regenold's 2026 {EU AI Act Q\&A Challenge}},
  author={Virgolin, Marco and Haszprunar, Thomas and Regenold, Maximilian},
  year={2026},
  howpublished = {arXiv preprint arXiv:2611.04546},
  doi = {https://doi.org/10.48550/arXiv.2611.04546},
  url = {http://arxiv.org/abs/2611.04546},
}
```
