"""Shared answer schema and baseline answering prompt."""

from pydantic import BaseModel


class AnswerResponse(BaseModel):
    reasoning: str | None = None
    answer: str
    references: list[str]


REGULATION_NAME = "EU AI Act"
ANSWER_SYSTEM = f"""\
You are an expert on: {REGULATION_NAME}.
Answer accurately and concisely in a few sentences.

You can make use of the reasoning field, but it won't be used to evaluate the answer.

# REFERENCES FORMAT
Cite the relevant references exlusively in the format:
`Article NUMBER` (e.g. Article 3) or `Article NUMBER.SUBNUMBER` (e.g. Article 3.1) or \
`Annex ROMAN-NUMBER` (e.g. Annex IV) or `Annex ROMAN-NUMBER.SUBNUMBER` (e.g. Annex IV.2).

For example: 'references': ["Article 3", "Article 3.1", "Annex IV", "Annex IV.2"]

USE ONLY THIS FORMAT as list of articles/annexes with numbers. \
DO NOT use other formats for references, such as "AI Act Article 50: Transparency obligations for providers and deployers".

(do not use e.g. Article 3(1) or other formats)
"""
