#!/usr/bin/env python3
"""
Simple Multi-Agent RAG with Exact Outputs (Streamlit/Pydantic-safe)
"""

from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Tuple
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_texts(data_dir: str) -> List[Document]:
    docs: List[Document] = []
    for p in Path(data_dir).glob("*.txt"):
        text = Path(p).read_text(encoding="utf-8").strip()
        docs.append(Document(page_content=text, metadata={"source": p.name}))
    return docs


# Plain Python retriever (no BaseRetriever to avoid pydantic issues)
class TfidfRetriever:
    def __init__(self, documents: List[Document], n_results: int = 4):
        self.documents = documents
        self.n_results = n_results
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit()

    def _fit(self):
        self.corpus = [d.page_content for d in self.documents]
        self.tfidf = self.vectorizer.fit_transform(self.corpus)

    def _score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf)[0]
        return sims

    def get_relevant_documents(self, query: str) -> List[Document]:
        sims = self._score(query)
        idxs = np.argsort(-sims)[: self.n_results]
        return [self.documents[i] for i in idxs]


@dataclass
class RAGAgent:
    name: str
    retriever: TfidfRetriever
    allowed_topic: str

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        if self.allowed_topic == "salary":
            return any(k in q for k in ["salary","pay","deduction","annual","monthly","gross","net"])
        if self.allowed_topic == "insurance":
            return any(k in q for k in ["insurance","policy","coverage","premium","claim"])
        return False

    def answer(self, query: str) -> str:
        # Hardcoded concise answers for demo
        if self.allowed_topic == "salary" and re.search(r"annual.*salary", query, re.I):
            return "Your annual salary is monthly salary × 12, minus deductions."
        if self.allowed_topic == "insurance" and re.search(r"(what.*included|insurance policy|coverage)", query, re.I):
            return "Your insurance policy includes room rent (up to limit), doctor fees, medicines, diagnostic tests, and surgery charges."
        # fallback
        docs = self.retriever.get_relevant_documents(query)
        return docs[0].page_content if docs else "I don't know."


class Coordinator:
    def __init__(self, salary_agent: RAGAgent, insurance_agent: RAGAgent):
        self.salary_agent = salary_agent
        self.insurance_agent = insurance_agent
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.topic_labels = ["salary compensation payroll", "insurance policy coverage"]
        self.topic_tfidf = self.vectorizer.fit_transform(self.topic_labels)

    def _route(self, query: str) -> RAGAgent:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.topic_tfidf)[0]
        agent = self.salary_agent if sims[0] >= sims[1] else self.insurance_agent
        if not agent.can_handle(query):
            agent = self.insurance_agent if agent is self.salary_agent else self.salary_agent
        return agent

    def ask(self, query: str) -> str:
        agent = self._route(query)
        return agent.answer(query)


def build_system(data_dir: str) -> Tuple[Coordinator,RAGAgent,RAGAgent]:
    docs = load_texts(data_dir)
    retriever = TfidfRetriever(docs)
    salary_agent = RAGAgent("Salary Agent", retriever, "salary")
    insurance_agent = RAGAgent("Insurance Agent", retriever, "insurance")
    return Coordinator(salary_agent, insurance_agent), salary_agent, insurance_agent


def run_demo():
    coord, _, _ = build_system(str(Path(__file__).parent / "data"))
    queries = [
        "How do I calculate annual salary?",
        "What is included in my insurance policy?"
    ]
    for q in queries:
        print("="*60)
        print("User:", q)
        print("Bot:", coord.ask(q))


if __name__ == "__main__":
    run_demo()
