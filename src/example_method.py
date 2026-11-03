"""Mock adapter: copy to my_method.py and connect your answering system."""

from contracts import AnswerResponse


def answer(history: list[dict[str, str]]) -> AnswerResponse:
    """Use the full history and return the final answer with structured citations."""
    return AnswerResponse(
        answer="This is a mock response. Replace it with your method's answer.",
        reasoning="This field can be used for the model to reason OUTSIDE of the answer and is not evaluated.",
        references=[],
    )
