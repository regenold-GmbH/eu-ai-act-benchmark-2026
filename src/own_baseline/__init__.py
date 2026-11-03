import os
from functools import cache
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pypdf import PdfReader

from contracts import ANSWER_SYSTEM, AnswerResponse

ROOT = Path(__file__).parents[2]
PDF = ROOT / "data/ai_act/eu-ai-act.pdf"
INDEX = ROOT / "data/ai_act/own_baseline_index"


@cache
def _components():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    if INDEX.exists():
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=INDEX), embed_model=embed_model
        )
    else:
        documents = [Document(text=page.extract_text() or "") for page in PdfReader(PDF).pages]
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX)
    return (
        index.as_retriever(similarity_top_k=5),
        OpenAI(
            model=os.getenv("OWN_BASELINE_MODEL", "gpt-5.5"),
            system_prompt=ANSWER_SYSTEM,
        ).as_structured_llm(AnswerResponse),
    )


def ask_own_baseline(history) -> AnswerResponse:
    retriever, llm = _components()
    question = history[-1]["content"]
    sources = "\n\n".join(node.text for node in retriever.retrieve(question))
    messages = [ChatMessage(role=item["role"], content=item["content"]) for item in history]
    messages[-1].content = f"Sources:\n{sources}\n\n{question}"
    return llm.chat(messages).raw
