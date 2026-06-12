import re
import logging
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean raw PDF text before chunking.
    Each step targets a specific type of noise — easy to add/remove rules.
    """
    if not text or not text.strip():
        return ""

    # 1. Fix hyphenated line breaks (PDF wraps long words)
    #    "impor-\ntant" → "important"
    text = re.sub(r"-\n", "", text)

    # 2. Remove page number patterns
    #    Matches: "Page 1", "Page 1 of 10", "1 | Page", "- 1 -"
    text = re.sub(r"(?i)(page\s+\d+(\s+of\s+\d+)?|\d+\s*\|\s*page|-\s*\d+\s*-)", "", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # 5. Remove copyright / confidentiality lines
    #    Matches lines like "© 2023 Acme Corp. All rights reserved."
    text = re.sub(r"(?i)©.*?(corp|inc|ltd|llc|company).*?\n", "", text)
    text = re.sub(r"(?i)(confidential|all rights reserved|proprietary).*?\n", "", text)

    # 6. Remove non-printable characters (garbage from bad PDF encoding)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # 7. Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """
    Apply cleaning to each page, drop pages that are empty after cleaning.
    WHY DROP EMPTY: An empty chunk wastes an embedding API call and
    adds a zero-information vector to your index.
    """
    cleaned = []
    dropped = 0

    for doc in docs:
        clean = clean_text(doc.page_content)
        if not clean:
            dropped += 1
            continue
        doc.page_content = clean
        cleaned.append(doc)

    logger.info(f"Cleaning done: {len(cleaned)} pages kept, {dropped} empty pages dropped.")
    return cleaned