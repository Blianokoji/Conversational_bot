"""
scrape/chunker.py
Splits cleaned text into overlapping chunks for the RAG knowledge base.
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into chunks of approximately *chunk_size* characters
    with *overlap* characters shared between consecutive chunks.

    Tries to break on sentence boundaries (`. `) when possible.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            # Last chunk — take everything remaining
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end
        # Look backwards from `end` for a period followed by whitespace
        boundary = text.rfind(". ", start + chunk_size // 2, end)
        if boundary != -1:
            end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
