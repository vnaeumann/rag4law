# src/router.py

from enum import Enum
from typing import Dict, Any, List


class Route(str, Enum):
    CASE_DOCS = "case_docs"
    LEGAL_DB = "legal_db"
    BOTH = "both"
    GENERAL = "general"


class QueryRouter:
    """
    Rule-based query router for Legal Case RAG.

    Routes user queries to:
    - case_docs: uploaded PDFs, affidavits, contracts, notices, evidence
    - legal_db: statutes, judgments, bare acts, precedents
    - both: when query needs both facts and law
    - general: general non-case/non-legal query
    """

    def __init__(self):
        self.legal_terms = [
            "section",
            "act",
            "ipc",
            "crpc",
            "cpc",
            "evidence act",
            "constitution",
            "article",
            "bare act",
            "statute",
            "provision",
            "legal provision",
            "case law",
            "precedent",
            "judgment",
            "judgement",
            "supreme court",
            "high court",
            "tribunal",
            "limitation act",
            "contract act",
            "specific relief act",
            "companies act",
            "arbitration act",
            "consumer protection act",
            "property act",
            "transfer of property",
            "criminal law",
            "civil law",
            "bail",
            "maintainability",
            "jurisdiction",
            "cause of action",
            "burden of proof",
            "mens rea",
            "prima facie",
        ]

        self.case_doc_terms = [
            "uploaded",
            "document",
            "documents",
            "pdf",
            "file",
            "affidavit",
            "contract",
            "agreement",
            "notice",
            "legal notice",
            "reply notice",
            "fir",
            "complaint",
            "plaint",
            "written statement",
            "petition",
            "counter",
            "rejoinder",
            "evidence",
            "exhibit",
            "annexure",
            "invoice",
            "receipt",
            "email",
            "letter",
            "order",
            "interim order",
            "hearing note",
            "transcript",
            "witness statement",
            "page",
            "clause",
            "paragraph",
            "para",
            "date",
            "timeline",
            "chronology",
            "contradiction",
            "inconsistency",
            "summary of facts",
            "facts of the case",
        ]

        self.both_terms = [
            "build the case",
            "argument",
            "arguments",
            "defence",
            "defense",
            "rebuttal",
            "legal strategy",
            "case strategy",
            "strength",
            "weakness",
            "weak points",
            "strong points",
            "issue",
            "issues",
            "frame issues",
            "relief",
            "prayer",
            "claim",
            "counter claim",
            "opponent argument",
            "petitioner's case",
            "respondent's case",
            "plaintiff's case",
            "defendant's case",
            "how to argue",
            "support my argument",
            "what law supports",
            "evidence matrix",
            "argument map",
            "case brief",
        ]

        self.general_terms = [
            "what is rag",
            "what is ai",
            "what is machine learning",
            "explain python",
            "write code",
            "debug",
            "streamlit",
            "faiss",
            "bm25",
            "embedding",
        ]

    def _contains_any(self, query: str, terms: List[str]) -> bool:
        q = query.lower()
        return any(term in q for term in terms)

    def route(self, query: str) -> Dict[str, Any]:
        """
        Return route decision with reason.

        Example output:
        {
            "route": "both",
            "reason": "Query appears to need both uploaded case facts and legal law/precedent.",
            "confidence": 0.85
        }
        """

        q = query.lower().strip()

        if not q:
            return {
                "route": Route.GENERAL.value,
                "reason": "Empty query.",
                "confidence": 0.0,
            }

        has_legal = self._contains_any(q, self.legal_terms)
        has_case_docs = self._contains_any(q, self.case_doc_terms)
        has_both = self._contains_any(q, self.both_terms)
        has_general = self._contains_any(q, self.general_terms)

        # Highest priority: mixed legal reasoning + case facts
        if has_both:
            return {
                "route": Route.BOTH.value,
                "reason": "Query likely needs both uploaded case facts and legal references.",
                "confidence": 0.85,
            }

        # Legal corpus only
        if has_legal and not has_case_docs:
            return {
                "route": Route.LEGAL_DB.value,
                "reason": "Query appears to ask about law, statute, legal provision, or precedent.",
                "confidence": 0.8,
            }

        # Case docs only
        if has_case_docs and not has_legal:
            return {
                "route": Route.CASE_DOCS.value,
                "reason": "Query appears to ask about uploaded case documents or evidence.",
                "confidence": 0.8,
            }

        # Both signals found
        if has_legal and has_case_docs:
            return {
                "route": Route.BOTH.value,
                "reason": "Query contains both legal-reference terms and uploaded-document terms.",
                "confidence": 0.9,
            }

        # General technical/non-legal query
        if has_general:
            return {
                "route": Route.GENERAL.value,
                "reason": "Query appears general or technical, not specifically about the case.",
                "confidence": 0.7,
            }

        # Default for your app:
        # Since this is a case-prep tool, assume case_docs.
        return {
            "route": Route.CASE_DOCS.value,
            "reason": "No strong legal/general signal found; defaulting to uploaded case documents.",
            "confidence": 0.55,
        }


# Convenience function
_router = QueryRouter()

def route_query(query: str) -> Dict[str, Any]:
    return _router.route(query)