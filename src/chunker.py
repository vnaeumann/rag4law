# src/chunker.py

import re


SPEAKER_PATTERN = re.compile(
    r"(?=(?:CHIEF JUSTICE|JUSTICE|MR\.|MS\.|MRS\.|DR\.|ADV\.|SOLICITOR GENERAL|ATTORNEY GENERAL|[A-Z][A-Z ]{3,}):)"
)


def detect_doc_type(text: str) -> str:
    t = text.lower()

    if "chief justice" in t or "hon'ble" in t or "court no" in t:
        return "transcript"

    if "affidavit" in t:
        return "affidavit"

    if "agreement" in t or "contract" in t or "clause" in t:
        return "contract"

    if "legal notice" in t or "notice" in t:
        return "notice"

    return "general_doc"


def word_chunk(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def paragraph_chunk(text, max_words=350):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    final_chunks = []

    for para in paras:
        words = para.split()

        if len(words) <= max_words:
            final_chunks.append(para)
        else:
            final_chunks.extend(
                word_chunk(para, chunk_size=max_words, overlap=80)
            )

    return final_chunks


def transcript_chunk(text, max_words=350):
    parts = SPEAKER_PATTERN.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 2:
        return paragraph_chunk(text, max_words=max_words)

    final_chunks = []
    buffer = []
    buffer_words = 0

    for part in parts:
        words = part.split()
        n = len(words)

        if buffer_words + n <= max_words:
            buffer.append(part)
            buffer_words += n
        else:
            if buffer:
                final_chunks.append(" ".join(buffer))

            if n > max_words:
                final_chunks.extend(
                    word_chunk(part, chunk_size=max_words, overlap=80)
                )
                buffer = []
                buffer_words = 0
            else:
                buffer = [part]
                buffer_words = n

    if buffer:
        final_chunks.append(" ".join(buffer))

    return final_chunks


def detect_legal_doc_type(text: str) -> str:
    t = text.lower()
    detected_type = detect_doc_type(text)

    if "section" in t or "act" in t:
        return "statute_or_bare_act"

    if "supreme court" in t or "judgment" in t or "judgement" in t:
        return "judgment"

    return detected_type


def build_chunks(pages, source_type="case_docs"):
    all_chunks = []

    for page in pages:
        text = page["text"]

        if source_type == "legal_db":
            doc_type = detect_legal_doc_type(text)
        else:
            doc_type = detect_doc_type(text)

        if doc_type == "transcript":
            chunks = transcript_chunk(text)
            chunk_type = "speaker_turn"
        else:
            chunks = paragraph_chunk(text)
            chunk_type = "paragraph"

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "doc_name": page["doc_name"],
                "page": page["page"],
                "chunk_id": f"{page['doc_name']}_p{page['page']}_c{i}",
                "source_type": source_type,
                "doc_type": doc_type,
                "chunk_type": chunk_type,
            })

    return all_chunks