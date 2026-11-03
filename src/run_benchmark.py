"""Evaluate a method on the EU AI Act Q&A benchmark.

Runs answering + LLM judge on each row of:
data/ai_act/2026-05-11-regenold_challenge_v01_01.xlsx

Validation sources:
- Correctness criteria via LLM judge (criteria extracted with regex)
- References via deterministic Article/Annex matching
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Callable

import litellm
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from contracts import ANSWER_SYSTEM, REGULATION_NAME, AnswerResponse
from latency_scoring import compute_response_speed_score
from ref_scoring import compute_ref_scores


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
INPUT_PATH = (
    ROOT
    / "data"
    / "ai_act"
    / "2026-05-11-regenold_challenge_v01_01.xlsx"
)
BASELINE_MODELS = {
    "web_search": os.getenv("WEB_SEARCH_MODEL", "openai/gpt-5.5"),
    "rag": os.getenv("OWN_BASELINE_MODEL", "gpt-5.5"),
}
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.4")


def llm_call_structured(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    temperature: float = 0.0,
    tools: list[dict] | None = None,
) -> BaseModel:

    if model.startswith("openai/"):
        # use response api
        resp = litellm.responses(
            model=model,
            input=[
                {"role": "system", "content": system},
            ]
            + messages,
            text_format=response_model,
            temperature=(
                temperature
                if model not in ["openai/gpt-5-search-api", "openai/gpt-5.5"]
                else None
            ),  # ignore temperature for models that do not support it
            tools=tools,
            stream=False,
        )
        return response_model.model_validate_json(resp.output_text)

    # use completion api
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
        ]
        + messages,
        response_format=response_model,
        temperature=(
            temperature
            if model not in ["openai/gpt-5-search-api", "openai/gpt-5.5"]
            else None
        ),  # ignore temperature for models that do not support it
        tools=tools,
        stream=False,
    )
    content = resp.choices[0].message.content

    if not content:
        raise ValueError("Empty structured response from model")
    return response_model.model_validate_json(content)


def extract_numbered_correctness_criteria(text: str) -> list[str]:
    """Extracts the correctness criteria from the textual format using a regex."""
    if not text:
        return []
    parts = re.findall(r"\(\d+\)\s*(.*?)(?=\s*\(\d+\)\s*|$)", text, flags=re.DOTALL)
    cleaned = [re.sub(r"\s+", " ", p).strip(" .;:\n\t\r") for p in parts]
    return [c for c in cleaned if c]



class CriterionJudgeResult(BaseModel):
    criterion: str
    consideration: str
    passed: bool


class JudgeCorrectnessResponse(BaseModel):
    criteria: list[CriterionJudgeResult]


class JudgeToneResponse(BaseModel):
    consideration: str
    appropriateness_passed: bool
    clarity_passed: bool


JUDGE_SYSTEM_CORRECTNESS = (
    f"You evaluate answers about: {REGULATION_NAME}.\n"
    "You consider the question and compare the proposed answer with the given criteria for correctness. "
    "\n\nHOW TO EVALUATE:\n"
    "- Provide a consideration/reasoning about each correctness criterion and, afterwards, mark each with pass/fail."
    "- Evaluate the substance of the answer, not exact wording (unless stated otherwise in the criterion). "
    "- If the answer includes intermediate reasoning steps, possibly mentioning previous answers, ignore those: evaluate only the final answer part. "
    "- NEVER question the correctness criteria themselves, just apply them as given. "
)

JUDGE_SYSTEM_TONE = f"""\
You consider the tone appropriateness and tone clarity of the language \
used by systems providing answers regarding: {REGULATION_NAME}.
You DO NOT evaluate the correctness of the answer content, just the tone.

### HOW TO EVALUATE:
- Provide a consideration/reasoning about the tone and clarity of the answer and, afterwards, mark each with pass/fail.
- For tone appropriateness, consider whether the language used in the answer is professional, formal, and suitable for a regulatory context. Mark with "pass/fail".
- For tone clarity, consider whether the language is clear, unambiguous, and not evasive. Mark with "pass/fail".

### EXAMPLES:
- Good tone and clarity (both pass):
    "Yes, an AI system is defined as a machine-based system that can, for a given set of human-defined objectives, generate outputs such as content, predictions, recommendations, or decisions influencing the environments they interact with. \
The definition is broad and includes various types of AI systems, regardless of their underlying technology or level of autonomy."
- Bad tone (fail tone, pass clarity):
    "Sure bro, the law got you covered in its definitions section: if it's a machine-based system that spits outputs for human-defined objectives, it's an AI system. Doesn't matter how it does it or how smart it is, it's all in."
- Bad clarity (pass tone, fail clarity):
    "The regulation addresses the matter of AI system definitions in a manner that could be interpreted as encompassing a wide range of technologies, typically generating recommendations but not necessarily only those, depending on the context and the specific characteristics of the system in question."
- Bad tone and clarity (both fail):
    "The regulation is like, you know, saying that if it's some machine thing that does stuff for humans, it's an AI system. But it's kinda vague and sounds unprofessional, so I wouldn't say it has a good tone or clarity."
"""


def ask_web_search(
    history: list[dict[str, str]],
) -> AnswerResponse:
    """Returns the answer and references for a given history (last item being question) using the LLM."""
    return llm_call_structured(
        model=BASELINE_MODELS["web_search"],
        system=ANSWER_SYSTEM,
        messages=history,
        response_model=AnswerResponse,
        tools=[{"type": "web_search_preview"}],
    )


def coerce_row_index(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean row index is not supported")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Unsupported row index type: {type(value).__name__}")


def evaluate_row(
    answering_llm_call: Callable,
    regulation: str,
    question: str,
    ground_truth_answer: str,
    correctness_criteria: str,
    references: str,
    no_judgements: int = 1,
    judge_temperature: float = 0.0,
    simulate_history: bool = False,
    simulate_pushback: bool = False,
) -> dict:
    history = []
    if simulate_history:
        convo_path = ROOT / "data" / "ai_act" / "9-steps-convo-nojson.json"
        with convo_path.open("r", encoding="utf-8") as f:
            convo = json.load(f)
        history = convo["messages"]

    latency_start_time = perf_counter()
    answer_response = answering_llm_call(
        history=history
        + [
            {
                "role": "user",
                "content": question,
            }
        ],
    )
    latency = perf_counter() - latency_start_time

    if simulate_pushback:
        pushback = (
            "I don't think this is correct. Perhaps your answer contains hallucinations."
            "\n\n"
            "(Briefly reason about whether something might indeed be incorrect, "
            "using the reasoning field. Then, provide a clear answer with the same format as before, "
            "as if I had just asked the same question anew: without mentioning the previous answer or the pushback.)\n\n"
            "Let's try again:\n" + question
        )
        history = history + [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": answer_response.answer,
            },
            {"role": "user", "content": pushback},
        ]
        latency_start_time = perf_counter()
        answer_response = answering_llm_call(
            history=history,
        )
        latency = perf_counter() - latency_start_time

    # compute references scores
    expected_refs = [
        ref.strip().lower() for ref in references.split(";") if ref.strip()
    ]
    ref_scores = compute_ref_scores(answer_response.references, expected_refs)

    # computes answer conciseness as ratio of words
    # between given answer and expected answer
    answer_conciseness_score = (
        len(answer_response.answer.split()) / len(ground_truth_answer.split())
        if ground_truth_answer
        else None
    )

    # compute correctness 
    criteria = extract_numbered_correctness_criteria(correctness_criteria)
    criteria_str = "\n".join(f"{i+1}. {a}" for i, a in enumerate(criteria))

    judge_corr_user_msg = (
        f"# Question:\n{question}\n\n"
        f"# Candidate answer:\n{answer_response.answer}\n\n"
        f"# Correctness criteria:\n{criteria_str}\n\n"
    )

    judgements_corr = []
    for _ in range(no_judgements):
        judgement = llm_call_structured(
            model=JUDGE_MODEL,
            system=JUDGE_SYSTEM_CORRECTNESS.format(
                regulation_name=regulation,
            ),
            messages=[{"role": "user", "content": judge_corr_user_msg}],
            response_model=JudgeCorrectnessResponse,
            temperature=judge_temperature,
        )
        judgements_corr.append(judgement.model_dump())

    # report if any discrepancy in correctness judgements
    discrepancies_correctness = []
    for criterion_idx, criterion in enumerate(criteria):
        criterion_judgements = [
            j["criteria"][criterion_idx]["passed"] for j in judgements_corr
        ]
        if len(set(criterion_judgements)) > 1:
            discrepancies_correctness.append(
                {
                    "criterion": criterion,
                    "judgements": criterion_judgements,
                }
            )

    # compute tone
    judge_tone_user_msg = f"# Candidate answer:\n{answer_response.answer}\n\n"

    judgements_tone = []
    for _ in range(no_judgements):
        judgement_tone = llm_call_structured(
            model=JUDGE_MODEL,
            system=JUDGE_SYSTEM_TONE.format(
                regulation_name=regulation,
            ),
            messages=[{"role": "user", "content": judge_tone_user_msg}],
            response_model=JudgeToneResponse,
            temperature=judge_temperature,
        )
        judgements_tone.append(judgement_tone.model_dump())

    # report if any discrepancy in tone judgements
    discrepancies_tone = []
    for criterion in ["appropriateness_passed", "clarity_passed"]:
        criterion_judgements = [j[criterion] for j in judgements_tone]
        if len(set(criterion_judgements)) > 1:
            discrepancies_tone.append(
                {
                    "criterion": criterion,
                    "judgements": criterion_judgements,
                }
            )

    return {
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "criteria": criteria,
        "candidate_reasoning": answer_response.reasoning,
        "candidate_answer": answer_response.answer,
        "candidate_references": answer_response.references,
        "answer_conciseness_score": answer_conciseness_score,
        "judgements_correctness": judgements_corr,
        "judgement_discrepancies_correctness": discrepancies_correctness,
        "judgements_tone": judgements_tone,
        "judgement_discrepancies_tone": discrepancies_tone,
        "references": references,
        "latency_seconds": latency,
        "response_speed_score": compute_response_speed_score(latency),
        **ref_scores,
    }


def run(
    limit: int | None,
    output_path: Path,
    debug: bool,
    answering_llm_call: Callable = ask_web_search,
    no_judgements: int = 1,
    judge_temperature: float = 0.0,
    specific_rows: list[int] | None = None,
    simulate_history: bool = False,
    simulate_pushback: bool = False,
    overwrite_output: bool = False,
    run_config: dict | None = None,
) -> None:
    if no_judgements < 1:
        raise ValueError("no_judgements must be at least 1")
    if judge_temperature < 0:
        raise ValueError("judge_temperature must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if no_judgements > 1 and judge_temperature == 0.0:
        raise ValueError(
            "If no_judgements > 1, judge_temperature should be > 0 to get diverse judgements"
        )

    if debug:
        print("Debug mode ON: litellm logs enabled")
        litellm._turn_on_debug()

    df = pd.read_excel(INPUT_PATH).fillna("")
    if limit is not None:
        df = df.head(limit)

    required_cols = [
        "Question",
        "Answer",
        "Correctness criteria",
        "References (Article, Annex)",
        "Regulation (with version or date)",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    completed_row_indices: set[int] = set()

    if output_path.exists() and not overwrite_output:
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

        if not isinstance(existing, list):
            raise ValueError(
                f"Existing output file {output_path} does not contain a JSON list. "
                "Use --overwrite_output to recreate it."
            )

        if run_config is not None and any(
            item.get("run_config") != run_config for item in existing
        ):
            raise ValueError(
                "Checkpoint configuration differs from this run. "
                "Choose another --output or use --overwrite_output."
            )
        results = existing
        completed_row_indices = {
            int(item["row_index"])
            for item in existing
            if isinstance(item, dict) and "row_index" in item
        }
        print(
            f"Resuming from existing report: {len(results)} saved entries, "
            f"{len(completed_row_indices)} unique completed row indices."
        )
    elif output_path.exists() and overwrite_output:
        print(
            f"Overwrite output enabled: existing report at {output_path} will be replaced."
        )

    if specific_rows is not None:
        unknown = set(specific_rows) - set(df.index)
        if unknown:
            raise ValueError(f"Selected rows are outside the dataset/limit: {sorted(unknown)}")
        df = df[df.index.isin(specific_rows)]

    rows_to_process = [idx for idx in df.index if idx not in completed_row_indices]
    if not rows_to_process:
        print(
            "No new rows to evaluate; all selected rows are already present in output."
        )
        print(f"Report available at {output_path}")
        return

    total_new = len(rows_to_process)
    processed_new = 0

    for idx, row in df.iterrows():
        if idx in completed_row_indices:
            continue

        idx_int = coerce_row_index(idx)

        processed_new += 1
        print(f"Evaluating row {idx_int + 1} ({processed_new}/{total_new} new rows)")
        result = evaluate_row(
            answering_llm_call=answering_llm_call,
            regulation=str(row["Regulation (with version or date)"]),
            question=str(row["Question"]),
            ground_truth_answer=str(row.get("Answer", "")),
            correctness_criteria=str(row["Correctness criteria"]),
            references=str(row["References (Article, Annex)"]),
            no_judgements=no_judgements,
            judge_temperature=judge_temperature,
            simulate_history=simulate_history,
            simulate_pushback=simulate_pushback,
        )
        result["row_index"] = idx_int
        if run_config is not None:
            result["run_config"] = run_config
        results.append(result)
        completed_row_indices.add(idx_int)

        checkpoint_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        checkpoint_path.replace(output_path)

    total = len(results)
    ok = sum(1 for r in results if "error" not in r)
    print(f"Done. {ok}/{total} rows evaluated.")
    print(f"Report written to {output_path}")


def dispatch_answering_llm_call(baseline: str) -> Callable:
    if baseline == "rag":
        from own_baseline import ask_own_baseline

        return ask_own_baseline
    return ask_web_search


def load_method(spec: str) -> Callable:
    module_name, separator, function_name = spec.partition(":")
    if not module_name or not separator or not function_name:
        raise ValueError("Use --method module:function, for example my_method:answer")
    method = getattr(importlib.import_module(module_name), function_name)
    if not callable(method):
        raise ValueError(f"{spec} is not callable")
    return method


def main() -> None:
    parser = argparse.ArgumentParser(description="EU AI Act benchmark: answer, judge, and save a resumable report")
    answerer = parser.add_mutually_exclusive_group(required=True)
    answerer.add_argument(
        "--method",
        help="Your answering function as module:function (for example my_method:answer)",
    )
    answerer.add_argument(
        "--baseline",
        choices=BASELINE_MODELS,
        help="Optional built-in answering baseline",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only first N rows"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSON path (default: results/<baseline or custom>[-hard].json)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with litellm logs"
    )
    parser.add_argument(
        "--no_judgements",
        type=int,
        default=1,
        help="Number of judgements to get per question",
    )
    parser.add_argument(
        "--judge_temperature",
        type=float,
        default=0.0,
        help="Temperature for judge LLM (higher for more diverse judgements if no_judgements > 1)",
    )
    parser.add_argument(
        "--specific_rows",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate only specific row indices (0-based)",
    )
    parser.add_argument(
        "--simulate_history",
        action="store_true",
        help="Whether to simulate a history of conversation (for testing answerer with conversation history)",
    )
    parser.add_argument(
        "--simulate_pushback",
        action="store_true",
        help="Whether to simulate a pushback from the user and a follow-up answer (for testing iterative answering)",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Replace the output instead of resuming completed rows",
    )
    args = parser.parse_args()

    if args.method:
        try:
            answering_llm_call = load_method(args.method)
        except (ImportError, AttributeError, ValueError) as error:
            parser.error(str(error))
    else:
        answering_llm_call = dispatch_answering_llm_call(args.baseline)
    suffix = (
        "-hard" if args.simulate_history and args.simulate_pushback
        else "-history" if args.simulate_history
        else "-pushback" if args.simulate_pushback
        else ""
    )
    output = args.output or ROOT / "results" / f"{args.baseline or 'custom'}{suffix}.json"
    run_config = {
        "judge_model": JUDGE_MODEL,
        "no_judgements": args.no_judgements,
        "judge_temperature": args.judge_temperature,
        "simulate_history": args.simulate_history,
        "simulate_pushback": args.simulate_pushback,
        "dataset_sha256": hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest(),
    }
    if args.method:
        run_config["method"] = args.method
    else:
        run_config["baseline"] = args.baseline
        run_config["answer_model"] = BASELINE_MODELS[args.baseline]
    if args.baseline == "rag":
        run_config["embedding_model"] = "text-embedding-3-small"
        run_config["retrieval_top_k"] = 5
        run_config["pdf_sha256"] = hashlib.sha256(
            (ROOT / "data/ai_act/eu-ai-act.pdf").read_bytes()
        ).hexdigest()
    if args.simulate_history:
        run_config["history_sha256"] = hashlib.sha256(
            (ROOT / "data/ai_act/9-steps-convo-nojson.json").read_bytes()
        ).hexdigest()

    run(
        answering_llm_call=answering_llm_call,
        limit=args.limit,
        output_path=output,
        debug=args.debug,
        no_judgements=args.no_judgements,
        judge_temperature=args.judge_temperature,
        specific_rows=args.specific_rows,
        simulate_history=args.simulate_history,
        simulate_pushback=args.simulate_pushback,
        overwrite_output=args.overwrite_output,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
